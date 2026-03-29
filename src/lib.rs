pub mod losses;
pub mod mahalanobis;
#[cfg(feature = "mining")]
pub mod mining;
pub mod similarity;

pub use losses::{
    circle_loss, contrastive_loss, cosine_embedding_loss, infonce_loss, lifted_structured_loss,
    matryoshka_loss, mnrl_loss, multi_similarity_loss, n_pairs_loss, supcon_loss, triplet_loss,
    triplet_loss_with_distance, DistanceFn, LossOutput,
};
pub use mahalanobis::{
    itml, kissme, learn_mahalanobis, learn_transform, lmnn, mahalanobis_distance, nca, transform,
    transform_batch, ItmlConfig, LmnnConfig, MahalanobisConfig, NcaConfig,
};
#[cfg(feature = "mining")]
pub use mining::{
    DistanceWeightedMiner, HardestMiner, InBatchMiner, MultiSimilarityMiner, RandomMiner,
    SemiHardMiner,
};
pub use similarity::{cosine_similarity, dot, euclidean_distance, l2_normalize, pairwise_cosine};
