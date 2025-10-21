from anndata import AnnData
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike
from numpyro.infer import SVI, Trace_ELBO
from patsy import dmatrix
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
from scipy.stats import norm
from tqdm import tqdm
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import re


# basic dataset representation to do minibatch sampling
class Dataset:
    def __init__(
        self, X: ArrayLike, design: ArrayLike, confounder_design: Optional[ArrayLike]
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
    ncells,
    X: jax.Array,
    design: jax.Array,
    confounder_design: Optional[jax.Array],
    negctrl_mask: jax.Array,
    mask: jax.Array,
):
    batch_size, ngenes = X.shape
    ncovariates = design.shape[1]

    σw = numpyro.sample("σw", dist.HalfCauchy(jnp.full(ncovariates, 1e-1)))
    w = numpyro.sample(
        "w", dist.Normal(jnp.zeros((ncovariates, ngenes)), jnp.expand_dims(σw, 1))
    )
    overdispersion = numpyro.sample("c", dist.HalfCauchy(jnp.full(ngenes, 10.0)))

    if confounder_design is not None:
        nconfounders = confounder_design.shape[1]
        b_mul = numpyro.sample(
            "b_mul", dist.Normal(jnp.zeros(nconfounders), jnp.full(nconfounders, 1.0))
        )
        b_add = numpyro.sample(
            "b_add",
            dist.LogNormal(jnp.zeros(nconfounders), jnp.full(nconfounders, 1.0)),
        )

    with (
        numpyro.plate("cells", ncells, subsample_size=batch_size, dim=-2),
        numpyro.handlers.mask(mask=mask),
        numpyro.plate("genes", ngenes, dim=-1),
    ):
        base_exp = design @ (w * negctrl_mask)
        if confounder_design is not None:
            nb_mean = jnp.exp(
                base_exp + confounder_design @ jnp.expand_dims(b_mul, 1)
            ) + confounder_design @ jnp.expand_dims(b_add, 1)  # [batch_size, ngenes]
        else:
            nb_mean = jnp.exp(base_exp)  # [batch_size, ngenes]

        numpyro.sample(
            "X",
            dist.NegativeBinomial2(
                mean=nb_mean,
                concentration=jnp.expand_dims(jnp.reciprocal(overdispersion), 0),
            ),
            obs=X,
        )


def guide(
    ncells,
    X,
    design,
    confounder_design: Optional[jax.Array],
    negctrl_mask,
    mask: jax.Array,
):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    σw_mu_q = numpyro.param("σw_mu_q", jnp.zeros(ncovariates))
    σw_sigma_q = numpyro.param(
        "σw_sigma_q", jnp.ones(ncovariates), constraint=dist.constraints.positive
    )
    numpyro.sample("σw", dist.LogNormal(σw_mu_q, σw_sigma_q))

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


def run_inference_vi(
    X, A, design, confounder_design, negctrl_mask, batch_size, nsamples
):
    optimizer = numpyro.optim.Adam(step_size=0.02)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(0)
    (params, _, _) = svi.run(
        key,
        nsamples,
        X=jnp.array(X, dtype=jnp.float32),
        A=A,
        design=jnp.array(design, dtype=jnp.float32),
        confounder_design=jnp.array(confounder_design, dtype=jnp.float32)
        if confounder_design is not None
        else None,
        negctrl_mask=jnp.array(negctrl_mask, dtype=jnp.float32),
        batch_size=batch_size,
    )
    params = {k: np.asarray(v) for k, v in params.items()}
    return params


class DEModel:
    def __init__(
        self,
        adata: AnnData,
        formula: str,
        confounder_formula: Optional[str] = None,
        negctrl_pat: str = "^NegControl",
        mask_negctrls: bool = False,
        model_segmentation_error: bool = False,
        max_edge_dist=15.0,
    ):
        """A model for performing differential expression analysis.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing gene expression data.
        formula : str
            The model formula for the fixed effects in R-style syntax.
        confounder_formula : Optional[str], default=None
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
        design = dmatrix(formula, adata.obs)
        if confounder_formula is not None:
            confounder_design = dmatrix(confounder_formula, adata.obs)
            # TODO: We might forcibly exclude an Intercept column if there is one
        else:
            confounder_design = None

        self.data = Dataset(adata.X, design, confounder_design)
        self.genes = adata.var_names
        self.design = design

        ncovariates = design.shape[1]
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

            edge_from = edge_from[distance_mask]
            edge_to = edge_to[distance_mask]

            edges = jnp.stack((edge_from, edge_to), axis=1)
            self.A = BCOO(
                (jnp.ones(len(edge_from)), edges),
                shape=(adata.shape[0], adata.shape[0]),
            )
        else:
            self.A = None

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
            negctrl_mask,
            mask,
        ):
            return svi.update(
                svi_state,
                ncells,
                X_batch,
                design_batch,
                confounder_batch,
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

    def de_results(self, covariate, minfc=1.5, credible=0.95):
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
        """
        j_intercept = self.get_design_column_idx("Intercept")
        intercept_posterior_mean = self.params["w_mu_q"][j_intercept, :]
        posterior_mean, lower_credibles, upper_credibles, down_probs, up_probs = (
            self._de_results(covariate, minfc, credible)
        )
        effect_size = np.maximum(
            np.clip(lower_credibles, 0, np.inf), -np.clip(upper_credibles, -np.inf, 0)
        )

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
        ).sort_values("abs_log2fc_bound", ascending=False)
