from anndata import AnnData
from jax.experimental.sparse import BCOO, bcoo_dot_general
from numpy.typing import ArrayLike
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
from scipy.stats import norm
from tqdm import tqdm
from typing import cast
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import numpyro
import numpyro.distributions as dist
import pandas as pd
import re


# basic dataset representation to do minibatch sampling
class Dataset:
    def __init__(
        self, X: ArrayLike, design: ArrayLike, confounder_design: None | ArrayLike
    ):
        self.X = X  # leave this as potentially a csr matrix
        self.design = np.asarray(design)
        self.confounder_design = (
            np.asarray(confounder_design) if confounder_design is not None else None
        )
        self.ncells = self.X.shape[0]
        self.rng = np.random.default_rng()
        assert self.X.shape[0] == self.ncells
        assert self.design.shape[0] == self.ncells
        if self.confounder_design is not None:
            assert self.confounder_design.shape[0] == self.ncells

    def __len__(self):
        return self.ncells

    def full_batch(self):
        X_batch = jnp.array(self.X)
        design_batch = jnp.array(self.design)
        confounder_design_batch = (
            jnp.array(self.confounder_design)
            if self.confounder_design is not None
            else None
        )
        mask = jnp.ones((self.X.shape[0], 1), dtype=bool)
        return X_batch, design_batch, confounder_design_batch, mask

    def iter_batches(self, batch_size: int):
        steps = (self.ncells + batch_size - 1) // batch_size  # ceil
        indices = self.rng.permutation(self.ncells)
        mask = np.ones((batch_size, 1), dtype=bool)

        for i in range(steps):
            start = i * batch_size
            end = min(start + batch_size, self.ncells)
            batch_indices = indices[start:end]

            X_batch = jnp.array(self.X[batch_indices, :])
            design_batch = jnp.array(self.design.take(batch_indices, axis=0))
            confounder_design_batch = (
                jnp.array(self.confounder_design.take(batch_indices, axis=0))
                if self.confounder_design is not None
                else None
            )

            if start + batch_size > self.ncells:
                padlen = batch_size - (self.ncells - start)
                mask[batch_size - padlen :, :] = False
                pad_width = ((0, padlen), (0, 0))
                X_batch = jnp.pad(X_batch, pad_width)
                design_batch = jnp.pad(design_batch, pad_width)
                confounder_design_batch = (
                    jnp.pad(confounder_design_batch, pad_width)
                    if confounder_design_batch is not None
                    else None
                )
                yield X_batch, design_batch, confounder_design_batch, jnp.array(mask)
            else:
                yield X_batch, design_batch, confounder_design_batch, jnp.array(mask)


# X: [ncells, ngenes]
# A: [ncells, ngenes] (sparse matrix)
# design: [ncells, ncovariates]
# negctrl_mask: [ncovariates, ngenes]
# batch_size: int
def model(
    ncells: int,
    X: jax.Array,
    design: jax.Array,
    confounder_design: jax.Array | None,
    diffusion_matrix: BCOO | None,
    diffusion_nneighbors: jax.Array | None,
    negctrl_mask: jax.Array,
    mask: jax.Array,
):
    batch_size, ngenes = X.shape
    ncovariates = design.shape[1]

    # "global" shrinkage that is per-gene
    σw = numpyro.sample("σw", dist.HalfCauchy(jnp.full((1, ngenes), 1.0)))
    # "local" shrinkage that is per-covariate and per-gene.
    λw = numpyro.sample("λw", dist.HalfCauchy(jnp.full((ncovariates - 1, ngenes), 1.0)))

    # We are assuming here that the design matrix as an intercept term and it's at index 0
    w = numpyro.sample(
        "w",
        dist.Normal(
            jnp.zeros((ncovariates, ngenes)),
            jnp.concatenate([jnp.full((1, ngenes), 5.0), λw * σw], axis=0),
        ),
    )

    overdispersion = numpyro.sample("c", dist.HalfCauchy(jnp.full(ngenes, 10.0)))

    # I guess this should be beta??
    diffusion_rate = numpyro.sample(
        name="d", fn=dist.Beta(jnp.full(ngenes, 10.0), jnp.full(ngenes, 1.0))
    )

    if confounder_design is not None:
        nconfounders = confounder_design.shape[1]
        b_mul = numpyro.sample(
            "b_mul", dist.Normal(jnp.zeros(nconfounders), jnp.full(nconfounders, 1.0))
        )
        b_add = numpyro.sample(
            "b_add",
            dist.LogNormal(jnp.zeros(nconfounders), jnp.full(nconfounders, 1.0)),
        )

    # gate_logits = numpyro.sample("g", dist.Normal(np.zeros(ngenes), 5.0))

    with (
        numpyro.plate("cells", ncells, subsample_size=batch_size, dim=-2),
        numpyro.handlers.mask(mask=mask),
        numpyro.plate("genes", ngenes, dim=-1),
    ):
        base_exp = design @ (w * negctrl_mask)
        if confounder_design is not None:
            # TODO: exclude cell weight if we decide to keep that
            nb_mean = jnp.exp(
                base_exp + confounder_design @ jnp.expand_dims(b_mul, 1)
            ) + confounder_design @ jnp.expand_dims(b_add, 1)  # [batch_size, ngenes]
        else:
            # nb_mean = jnp.exp(
            #     base_exp + jnp.expand_dims(cell_weight, 1)
            # )  # [batch_size, ngenes]
            nb_mean = jnp.exp(base_exp)  # [batch_size, ngenes]

        if diffusion_matrix is not None and diffusion_nneighbors is not None:
            # simplest model, we should really be doing this at the level of poisson rate
            nb_mean = diffusion_matrix @ (diffusion_rate * nb_mean) + nb_mean

            # We could just model some additional per-cell rate that we add onto nb_mean I guess?

        # likelihood = dist.NegativeBinomial2(
        #     mean=nb_mean,
        #     concentration=jnp.expand_dims(jnp.reciprocal(overdispersion), 0),
        # )
        #

        # jax.debug.print("mean(gate_logits): {}", gate_logits.mean())

        # likelihood_dist = dist.ZeroInflatedNegativeBinomial2(
        #     mean=nb_mean,
        #     concentration=jnp.expand_dims(jnp.reciprocal(overdispersion), 0),
        #     gate_logits=jnp.expand_dims(gate_logits, 0),
        # )

        likelihood_dist = dist.NegativeBinomial2(
            mean=nb_mean,
            concentration=jnp.expand_dims(jnp.reciprocal(overdispersion), 0),
        )

        numpyro.sample(
            "X",
            likelihood_dist,
            obs=X,
        )

        # Fitting against all zeros, as a test.
        # numpyro.sample(
        #     "X",
        #     likelihood_dist,
        #     obs=np.zeros(X.shape),
        # )


def guide(
    ncells,
    X,
    design,
    confounder_design: jax.Array | None,
    _diffusion_matrix: BCOO | None,
    _diffusion_nneighbors: jax.Array | None,
    negctrl_mask,
    mask: jax.Array,
):
    _ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    σw_mu_q = numpyro.param("σw_mu_q", jnp.zeros((1, ngenes)))
    σw_sigma_q = numpyro.param(
        "σw_sigma_q",
        jnp.ones((1, ngenes)),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("σw", dist.LogNormal(σw_mu_q, σw_sigma_q))

    λw_mu_q = numpyro.param("λw_mu_q", jnp.zeros((ncovariates - 1, ngenes)))
    λw_sigma_q = numpyro.param(
        "λw_sigma_q",
        jnp.ones((ncovariates - 1, ngenes)),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("λw", dist.LogNormal(λw_mu_q, λw_sigma_q))

    # b_mu_q = numpyro.param("b_mu_q", jnp.full((1, ngenes), 0.0))
    # b_sigma_q = numpyro.param(
    #     "b_sigma_q",
    #     jnp.ones((1, ngenes)),
    #     constraint=dist.constraints.positive,
    # )
    # numpyro.sample("b", dist.Normal(b_mu_q, b_sigma_q))

    w_mu_q = numpyro.param("w_mu_q", jnp.zeros((ncovariates, ngenes)))
    w_sigma_q = numpyro.param(
        "w_sigma_q",
        jnp.ones((ncovariates, ngenes)),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("w", dist.Normal(w_mu_q, w_sigma_q))

    c_mu_q = numpyro.param("c_mu_q", jnp.full(ngenes, 0.0))
    c_sigma_q = numpyro.param(
        "c_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive
    )
    numpyro.sample("c", dist.LogNormal(c_mu_q, c_sigma_q))

    d_log_alpha_q = numpyro.param(
        "d_alpha_q",
        init_value=np.full(1, 1e-1),
        constraint=dist.constraints.positive,
    )
    d_log_beta_q = numpyro.param(
        "d_beta_q",
        init_value=np.full(ngenes, 1e-1),
        constraint=dist.constraints.positive,
    )
    numpyro.sample("d", dist.Beta(jnp.exp(d_log_alpha_q), jnp.exp(d_log_beta_q)))

    u_mu_q = numpyro.param("u_mu_q", jnp.zeros(ncells))
    u_sigma_q = numpyro.param(
        "u_sigma_q", jnp.ones(ncells), constraint=dist.constraints.positive
    )
    numpyro.sample("u", dist.Normal(u_mu_q, u_sigma_q))

    g_mu_q = numpyro.param("g_mu_q", jnp.zeros(ngenes))
    g_sigma_q = numpyro.param(
        "g_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive
    )
    numpyro.sample("g", dist.Normal(g_mu_q, g_sigma_q))

    if confounder_design is not None:
        nconfounders = confounder_design.shape[1]
        b_mul_mu_q = numpyro.param("b_mul_mu_q", jnp.zeros(nconfounders))
        b_mul_sigma_q = numpyro.param(
            "b_mul_sigma_q",
            jnp.ones(nconfounders),
            constraint=dist.constraints.positive,
        )
        numpyro.sample("b_mul", dist.Normal(b_mul_mu_q, b_mul_sigma_q))

        b_add_mu_q = numpyro.param("b_add_mu_q", jnp.full(nconfounders, 1e-2))
        b_add_sigma_q = numpyro.param(
            "b_add_sigma_q",
            jnp.ones(nconfounders),
            constraint=dist.constraints.positive,
        )
        numpyro.sample("b_add", dist.LogNormal(b_add_mu_q, b_add_sigma_q))


class DEModel:
    params: None | dict[str, NDArray[np.float32]]
    data: Dataset
    genes: NDArray[np.str_]
    design: DesignMatrix
    negctrl_mask: NDArray[np.bool_]
    A: None | BCOO

    def __init__(
        self,
        adata: AnnData,
        formula: str,
        confounder_formula: None | str = None,
        negctrl_pat: str = "^NegControl",
        mask_negctrls: bool = False,
        model_segmentation_error: bool = False,
        max_edge_dist=25.0,
    ):
        """A model for performing differential expression analysis.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing gene expression data.
        formula : str
            The model formula for the fixed effects in R-style syntax.
        confounder_formula : None | str, default=None
            The model formula for confounder variables in R-style syntax.
        negctrl_pat : str, default="^NegControl"
            Regular expression pattern to identify negative control genes.
        mask_negctrls : bool, default=False
            Whether to mask out regression coefficients for negative control genes.
        model_segmentation_error : bool, default=False
            Whether to model segmentation errors between adjacent cells.
        max_edge_dist : float, default=15.0
            Maximum distance between cells to be considered adjacent.
        """
        self.params = None
        design = cast(DesignMatrix, dmatrix(formula, adata.obs))

        print(type(design))
        if confounder_formula is not None:
            confounder_design = dmatrix(confounder_formula, adata.obs)
            # TODO: We might forcibly exclude an Intercept column if there is one
        else:
            confounder_design = None

        # TODO: deal with adata.X being potentially sparse, non-integer, etc.
        self.data = Dataset(adata.X, design, confounder_design)
        self.genes = np.asarray(adata.var_names)
        self.design = design

        ncovariates = int(design.shape[1])
        ngenes = len(self.genes)

        if mask_negctrls:
            negctrl_mask = np.array(
                [re.match(negctrl_pat, gene) is None for gene in adata.var_names],
                dtype=bool,
            )
            nnegctrls = np.sum(~negctrl_mask)
            print(f"{nnegctrls} negative controls")
            negctrl_mask = np.repeat(
                np.expand_dims(negctrl_mask, 0), ncovariates, axis=0
            )  # mask out regression coefficients for negative controls

            if "Intercept" in design.design_info.column_names:
                negctrl_mask[0, :] = True  # allow intercepts for negative controls

            self.negctrl_mask = negctrl_mask
        else:
            self.negctrl_mask = np.ones((ncovariates, ngenes), dtype=bool)

        if model_segmentation_error:
            if "spatial" not in adata.obsm:
                raise ValueError(
                    "Spatial information is required for segmentation error modeling."
                )

            xys = np.asarray(adata.obsm["spatial"])
            tri = Delaunay(xys)
            tri_indptr, tri_indices = tri.vertex_neighbor_vertices

            edge_from = []
            edge_to = []

            for i in range(adata.shape[0]):
                for j in tri_indices[tri_indptr[i] : tri_indptr[i + 1]]:
                    if i != j:
                        edge_from.append(i)
                        edge_to.append(j)

            edge_from = np.array(edge_from, dtype=np.int32)
            edge_to = np.array(edge_to, dtype=np.int32)

            distances = np.linalg.norm(xys[edge_from] - xys[edge_to], axis=1)
            distance_mask = distances <= max_edge_dist

            nneighbors = np.zeros(adata.shape[0], dtype=np.float32)
            for i in edge_from[distance_mask]:
                nneighbors[i] += 1

            edge_from = edge_from[distance_mask]
            edge_to = edge_to[distance_mask]

            edges = jnp.stack((edge_from, edge_to), axis=1)
            self.A = BCOO(
                (jnp.ones(len(edge_from)), edges),
                shape=(adata.shape[0], adata.shape[0]),
            )
            self.nneighbors = jnp.array(nneighbors)
        else:
            self.A = None
            self.nneighbors = None

    @property
    def design_matrix(self):
        return self.data.design

    @property
    def confounder_matrix(self):
        return self.data.confounder_design

    def _check_is_fit(self) -> None:
        if self.params is None:
            raise ValueError(
                "Model must be fit before running inference. Call .fit() first"
            )

    def get_design_column_idx(self, col: str | int) -> int:
        """Get a design matrix column index by name."""
        if isinstance(col, int):
            return col
        elif not isinstance(col, str):
            raise ValueError("Column must be string.")
        if col not in self.design.design_info.column_names:
            raise ValueError(f"Column {col} not found in design matrix.")
        return self.design.design_info.column_names.index(col)

    def _fit_unbatched(self, nepochs: int, platform: str):
        numpyro.set_platform(platform)
        negctrl_mask = jnp.array(self.negctrl_mask)
        optimizer = numpyro.optim.Adam(step_size=0.02)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        key = jax.random.PRNGKey(0)
        ncells = len(self.data)

        def update_fn(
            svi_state,
            ncells,
            X_batch,
            design_batch,
            confounder_batch,
            diffusion_matrix,
            diffusion_nneighbors,
            negctrl_mask,
            mask,
        ):
            return svi.update(
                svi_state,
                ncells,
                X_batch,
                design_batch,
                confounder_batch,
                diffusion_matrix,
                diffusion_nneighbors,
                negctrl_mask,
                mask,
            )

        svi_state = None
        prog = tqdm(range(nepochs), desc="Training epochs")
        X_batch, design_batch, confounder_batch, mask = self.data.full_batch()
        for epoch in prog:
            # OK, not doing batches. Have to pass the whole thing in one go.

            if svi_state is None:
                svi_state = svi.init(
                    key,
                    ncells,
                    X_batch,
                    design_batch,
                    confounder_batch,
                    self.A,
                    self.nneighbors,
                    negctrl_mask,
                    mask,
                )

            svi_state, loss = jax.jit(update_fn, static_argnums=1)(
                svi_state,
                ncells,
                X_batch,
                design_batch,
                confounder_batch,
                self.A,
                self.nneighbors,
                negctrl_mask,
                mask,
            )
            prog.set_postfix({"loss": f"{loss:.4f}"})

        params = svi.get_params(svi_state)
        self.params = {k: np.asarray(v) for k, v in params.items()}

    def fit(
        self, nepochs: int = 400, batch_size: int | None = 32768, platform: str = "gpu"
    ):
        """Fit parameters to the dataset.

        Parameters
        ----------
        nsamples : int, default=6000
            Number of samples to draw during variational inference.
        batch_size : int, default=1024
            Mini-batch size for stochastic variational inference.

        Returns
        -------
        None
            Model parameters are stored in the self.params attribute.
        """
        ncells = len(self.data)

        if batch_size is None or batch_size > ncells or self.A is not None:
            print("Using unbatched training. This may run out of memory!")
            return self._fit_unbatched(nepochs, platform)

        numpyro.set_platform(platform)
        negctrl_mask = jnp.array(self.negctrl_mask)

        optimizer = numpyro.optim.Adam(step_size=0.02)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        key = jax.random.PRNGKey(0)

        def update_fn(
            svi_state,
            ncells,
            X_batch,
            design_batch,
            confounder_batch,
            negctrl_mask,
            mask,
        ):
            return svi.update(
                svi_state,
                ncells,
                X_batch,
                design_batch,
                confounder_batch,
                None,
                None,
                negctrl_mask,
                mask,
            )

        svi_state = None
        prog = tqdm(range(nepochs), desc="Training epochs")
        for epoch in prog:
            epoch_loss = 0.0
            for X_batch, design_batch, confounder_batch, mask in self.data.iter_batches(
                batch_size
            ):
                if svi_state is None:
                    svi_state = svi.init(
                        key,
                        ncells,
                        X_batch,
                        design_batch,
                        confounder_batch,
                        negctrl_mask,
                        mask,
                    )

                svi_state, loss = jax.jit(update_fn, static_argnums=1)(
                    svi_state,
                    ncells,
                    X_batch,
                    design_batch,
                    confounder_batch,
                    negctrl_mask,
                    mask,
                )
                epoch_loss += loss
            prog.set_postfix({"loss": f"{epoch_loss:.4f}"})

        params = svi.get_params(svi_state)
        self.params = {k: np.asarray(v) for k, v in params.items()}

    def _estimate_abs_effect_size(
        self, covariate, min_shift, credible, nsamples: int = 1000
    ):
        print("estimate_abs_effect_size")
        self._check_is_fit()
        j = self.get_design_column_idx(covariate)

        design = self.data.design  # [ncells, ncovariates]

        w_mu = self.params["w_mu_q"]  # [ncovariates, ngenes]
        w_sigma = self.params["w_sigma_q"]  # [ncovariates, ngenes]

        ngenes = w_mu.shape[1]

        design_sansj = np.delete(design, j, 1)  # [ncells, ncovariates - 1]
        w_mu_sansj = np.delete(w_mu, j, 0)  # [ncovariates - 1, ngenes]
        w_sigma_sansj = np.delete(w_sigma, j, 0)  # [ncovariates - 1, ngenes]

        cell_mu_sansj = jnp.array(design_sansj @ w_mu_sansj)
        cell_sigma_sansj = jnp.array(np.sqrt(design_sansj @ np.square(w_sigma_sansj)))

        design_j = design[:, j : j + 1]  # [ncells, 1]
        w_mu_j = w_mu[j : j + 1, :]  # [1, ngenes]
        w_sigma_j = w_sigma[j : j + 1, :]  # [1, ngenes]

        cell_mu_j = jnp.array(design_j @ w_mu_j)
        cell_sigma_j = jnp.array(np.sqrt(design_j @ np.square(w_sigma_j)))

        abs_effect_mean = np.zeros(ngenes, dtype=np.float32)

        def sample_diff(key, cell_mu_sansj, cell_sigma_sansj, cell_mu_j, cell_sigma_j):
            log_expr0 = (
                jax.random.normal(key, cell_mu_sansj.shape) * cell_sigma_sansj
                + cell_mu_sansj
            )

            log_expr1 = (
                log_expr0
                + jax.random.normal(key, cell_mu_j.shape) * cell_sigma_j
                + cell_mu_j
            )

            abs_effect_sample = (
                jnp.exp(log_expr0 + log_expr1) - jnp.exp(log_expr0)
            ).mean(axis=0)

            return abs_effect_sample

        key = jax.random.PRNGKey(0)
        for sample_num in range(nsamples):
            print(sample_num)
            key, sample_key = jax.random.split(key)

            abs_effect_mean += np.array(
                jax.jit(sample_diff)(
                    key,
                    cell_mu_sansj,
                    cell_sigma_sansj,
                    cell_mu_j,
                    cell_sigma_j,
                )
            )

        abs_effect_mean /= nsamples

        print(abs_effect_mean)

        # TODO: Do we have to sample?

    def _de_results(self, covariate: str | int, minfc, credible):
        self._check_is_fit()
        log_minfc = np.log(minfc)
        j = self.get_design_column_idx(covariate)
        ngenes = self.params["w_mu_q"].shape[1]

        posterior_mean = self.params["w_mu_q"][j, :]
        lower_credibles = np.zeros(ngenes, dtype=np.float32)
        upper_credibles = np.zeros(ngenes, dtype=np.float32)
        down_probs = np.zeros(ngenes, dtype=np.float32)
        up_probs = np.zeros(ngenes, dtype=np.float32)

        for j, (μ, σ) in enumerate(
            zip(self.params["w_mu_q"][j, :], self.params["w_sigma_q"][j, :])
        ):
            lower_credibles[j], upper_credibles[j] = norm.interval(
                credible, loc=μ, scale=σ
            )
            down_probs[j] = norm.cdf(-log_minfc, loc=μ, scale=σ)
            up_probs[j] = 1 - norm.cdf(log_minfc, loc=μ, scale=σ)

        return posterior_mean, lower_credibles, upper_credibles, down_probs, up_probs

    def de_results(
        self, covariate: str, minfc: float = 1.5, credible: float = 0.95
    ) -> pd.DataFrame:
        """Return differential expression analysis results.

        Parameters
        ----------
        covariate : str or int
            The covariate name or index to test for differential expression.
        minfc : float, default=1.5
            Minimum fold change threshold.
        credible : float, default=0.95
            Credible interval width.

        Returns
        -------
        pd.DataFrame
            DataFrame containing differential expression analysis results with columns:
            - gene: Gene name
            - log10_base_mean: Log10 of base expression level
            - log2fc: Log2 fold change
            - log2fc_lower_credible: Lower bound of credible interval for log2fc
            - log2fc_upper_credible: Upper bound of credible interval for log2fc
            - de_down_prob: Probability of downregulation exceeding minfc
            - de_up_prob: Probability of upregulation exceeding minfc
            - de_prob: Total probability of differential expression
            - abs_log2fc_bound: Lower bound on the absolute log fold change
        """
        j_intercept = self.get_design_column_idx("Intercept")
        intercept_posterior_mean = self.params["w_mu_q"][j_intercept, :]

        # TODO: figure out what this reports
        self._estimate_abs_effect_size(covariate, credible, 0.1)

        posterior_mean, lower_credibles, upper_credibles, down_probs, up_probs = (
            self._de_results(covariate, minfc, credible)
        )
        effect_size = np.maximum(
            np.clip(lower_credibles, 0, np.inf), -np.clip(upper_credibles, -np.inf, 0)
        )

        # I think what we really need to estimate quantiles for absolute effect size, but for that
        # we have to consider the other coefficients.

        return pd.DataFrame(
            dict(
                gene=self.genes,
                log10_base_mean=intercept_posterior_mean / np.log(10),
                log2fc=posterior_mean / np.log(2),
                log2fc_lower_credible=lower_credibles / np.log(2),
                log2fc_upper_credible=upper_credibles / np.log(2),
                de_down_prob=down_probs,
                de_up_prob=up_probs,
                de_prob=down_probs + up_probs,
                abs_log2fc_bound=effect_size / np.log(2),
            )
        )

        # Probably a bad idea, so we can compare to the adata without joining.
        # .sort_values("abs_log2fc_bound", ascending=False)
