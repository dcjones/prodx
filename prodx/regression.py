
from anndata import AnnData
from jax._src.typing import DType
from numpyro.infer import SVI, Trace_ELBO
from patsy import dmatrix
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

def model(X, Xneighbors, design):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w = numpyro.sample("w", dist.Cauchy(jnp.zeros((ncovariates, ngenes)), jnp.full((ncovariates, ngenes), 1e-1)))
    c = numpyro.sample("c", dist.HalfCauchy(jnp.full(ngenes, 1.0)))

    nb_mean = jnp.exp(design@w)

    if Xneighbors is not None:
        b_diffusion = numpyro.sample(
            "b",
            dist.HalfCauchy(jnp.full(ngenes, 1e-1)))
        nb_mean += b_diffusion * Xneighbors

    numpyro.sample(
        "X",
        dist.NegativeBinomial2(
            mean=nb_mean,
            concentration=jnp.repeat(jnp.expand_dims(c, 0), ncells, axis=0),
        ),
        obs=X)

def guide(X, Xneighbors, design):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w_mu_q = numpyro.param("w_mu_q", jnp.zeros((ncovariates, ngenes)))
    w_sigma_q = numpyro.param("w_sigma_q", jnp.ones((ncovariates, ngenes)), constraint=dist.constraints.positive)
    numpyro.sample("w", dist.Normal(w_mu_q, w_sigma_q))

    if Xneighbors is not None:
        b_mu_q = numpyro.param("b_mu_q", jnp.zeros(ngenes))
        b_sigma_q = numpyro.param("b_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
        numpyro.sample("b", dist.LogNormal(b_mu_q, b_sigma_q))

    c_mu_q = numpyro.param("c_mu_q", jnp.zeros(ngenes))
    c_sigma_q = numpyro.param("c_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
    numpyro.sample("c", dist.LogNormal(c_mu_q, c_sigma_q))


def run_inference_vi(X, Xneighbors, design, nsamples):
    optimizer = numpyro.optim.Adam(step_size=0.02)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(0)
    (params, _, _) = svi.run(
        key, nsamples,
        X=jnp.array(X, dtype=jnp.float32),
        Xneighbors=jnp.array(Xneighbors, dtype=jnp.float32) if Xneighbors is not None else None,
        design=jnp.array(design, dtype=jnp.float32))
    params = {
        k: np.asarray(v) for k, v in params.items()
    }
    return params

def merge_inferred_params(param_chunks: list):
    params = {
        "w_mu_q": np.concatenate([chunk["w_mu_q"] for chunk in param_chunks], axis=1),
        "w_sigma_q": np.concatenate([chunk["w_sigma_q"] for chunk in param_chunks], axis=1),
        "c_mu_q": np.concatenate([chunk["c_mu_q"] for chunk in param_chunks], axis=0),
        "c_sigma_q": np.concatenate([chunk["c_sigma_q"] for chunk in param_chunks], axis=0),
    }

    if "b_mu_q" in param_chunks[0]:
        params["b_mu_q"] = np.concatenate([chunk["b_mu_q"] for chunk in param_chunks], axis=0)
    if "b_sigma_q" in param_chunks[0]:
        params["b_sigma_q"] = np.concatenate([chunk["b_sigma_q"] for chunk in param_chunks], axis=0)

    return params

class DEModel:
    def __init__(self, adata: AnnData, formula: str, model_segmentation_error: bool=False, max_edge_dist=15.0):
        self.adata = adata
        self.design = dmatrix(formula, adata.obs)

        if model_segmentation_error:
            if "spatial" not in adata.obsm:
                raise ValueError("Spatial information is required for segmentation error modeling.")

            xys = np.asarray(adata.obsm["spatial"])
            tri = Delaunay(xys)
            tri_indptr, tri_indices = tri.vertex_neighbor_vertices

            edge_from = []
            edge_to = []
            for i in range(adata.shape[0]):
                for j in tri_indices[tri_indptr[i]:tri_indptr[i+1]]:
                    if i != j:
                        edge_from.append(i)
                        edge_to.append(j)

            edge_from = np.array(edge_from, dtype=np.int32)
            edge_to = np.array(edge_to, dtype=np.int32)

            distances = np.linalg.norm(xys[edge_from] - xys[edge_to], axis=1)
            distance_mask = distances <= max_edge_dist

            edge_from = edge_from[distance_mask]
            edge_to = edge_to[distance_mask]

            A = coo_matrix((np.ones(len(edge_from)), (edge_from, edge_to)), shape=(adata.shape[0], adata.shape[0]))
            # TODO: do we normalize this, or just assume more neighbors leads to more diffusion?
            #
            self.Xneighbors = A @ adata.X
            print(self.Xneighbors)
        else:
            self.Xneighbors = None

    def fit(self, nsamples=2000, chunksize=64):
        # I think we have to do this in chunks!
        param_chunks = []
        ngenes = self.adata.shape[1]
        for j in range(0, ngenes, chunksize):
            # Calculate the actual size of this chunk (might be smaller for last chunk)
            current_chunksize = min(chunksize, ngenes - j)
            print(current_chunksize)
            param_chunks.append(run_inference_vi(
                self.adata.X[:,j:j+current_chunksize],
                self.Xneighbors[:,j:j+current_chunksize] if self.Xneighbors is not None else None,
                self.design, nsamples=nsamples))

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
