
from anndata import AnnData
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO
from patsy import dmatrix
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

def model(X, design):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w = numpyro.sample("w", dist.Cauchy(jnp.zeros((ncovariates, ngenes)), jnp.full((ncovariates, ngenes), 1e-1)))
    c = numpyro.sample("c", dist.HalfCauchy(jnp.full(ngenes, 1.0)))

    numpyro.sample(
        "X",
        dist.NegativeBinomial2(
            mean=jnp.exp(design@w),
            concentration=jnp.repeat(jnp.expand_dims(c, 0), ncells, axis=0),
        ),
        obs=X)

def guide(X, design):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w_mu_q = numpyro.param("w_mu_q", jnp.zeros((ncovariates, ngenes)))
    w_sigma_q = numpyro.param("w_sigma_q", jnp.ones((ncovariates, ngenes)), constraint=dist.constraints.positive)
    numpyro.sample("w", dist.Normal(w_mu_q, w_sigma_q))

    c_mu_q = numpyro.param("c_mu_q", jnp.zeros(ngenes))
    c_sigma_q = numpyro.param("c_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
    numpyro.sample("c", dist.LogNormal(c_mu_q, c_sigma_q))


# I think this is just too slow to be useful in most cases
# def run_inference_mcmc(X, design, nsamples, nburnin):
#     key = jax.random.PRNGKey(0)
#     kernel = NUTS(model)
#     mcmc = MCMC(kernel, num_warmup=nburnin, num_samples=nsamples)

#     mcmc.run(
#         key,
#         X=jnp.array(X),
#         design=jnp.array(design))

#     # TODO: return some meaningful results
#     mcmc.print_summary()


def run_inference_vi(X, design, nsamples):
    optimizer = numpyro.optim.Adam(step_size=0.02)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(0)
    (params, _, _) = svi.run(key, nsamples, X=jnp.array(X, dtype=jnp.float32), design=jnp.array(design, dtype=jnp.float32))
    params = {
        k: np.asarray(v) for k, v in params.items()
    }
    return params

def merge_inferred_params(param_chunks: list):
    return {
        "w_mu_q": np.concatenate([chunk["w_mu_q"] for chunk in param_chunks], axis=1),
        "w_sigma_q": np.concatenate([chunk["w_sigma_q"] for chunk in param_chunks], axis=1),
        "c_mu_q": np.concatenate([chunk["c_mu_q"] for chunk in param_chunks], axis=0),
        "c_sigma_q": np.concatenate([chunk["c_sigma_q"] for chunk in param_chunks], axis=0),
    }

class DEModel:
    def __init__(self, adata: AnnData, formula: str):
        self.adata = adata
        self.design = dmatrix(formula, adata.obs)

    def fit(self, nsamples=2000, chunksize=64):
        # I think we have to do this in chunks!
        param_chunks = []
        ngenes = self.adata.shape[1]
        for j in range(0, ngenes, chunksize):
            # Calculate the actual size of this chunk (might be smaller for last chunk)
            current_chunksize = min(chunksize, ngenes - j)
            print(current_chunksize)
            param_chunks.append(run_inference_vi(self.adata.X[:,j:j+current_chunksize], self.design, nsamples=nsamples))

        params = merge_inferred_params(param_chunks)

        # posterior mean coefficient estimates (should we convert these to log2?)
        coefs = pd.DataFrame(
            np.asarray(params["w_mu_q"]).transpose(),
            columns=self.design.design_info.column_names,
            index=self.adata.var_names,
        )

        sds = pd.DataFrame(
            np.asarray(params["w_sigma_q"]).transpose(),
            columns=self.design.design_info.column_names,
            index=self.adata.var_names,
        )

        return coefs, sds
