pub mod losses;
pub mod mahalanobis;
#[cfg(feature = "mining")]
pub mod mining;
pub mod similarity;

#[cfg(feature = "candle")]
pub mod candle_losses;

pub use losses::{
    CrossBatchMemory, DistanceFn, LossOutput, arcface_loss, circle_loss, contrastive_loss,
    cosine_embedding_loss, infonce_loss, lifted_structured_loss, matryoshka_loss, mnrl_loss,
    multi_similarity_loss, n_pairs_loss, proxy_anchor_loss, supcon_loss, triplet_loss,
    triplet_loss_with_distance, vicreg_loss,
};
pub use mahalanobis::{
    ItmlConfig, LmnnConfig, MahalanobisConfig, NcaConfig, itml, kissme, learn_mahalanobis,
    learn_transform, lmnn, mahalanobis_distance, nca, transform, transform_batch,
};
#[cfg(feature = "mining")]
pub use mining::{
    DistanceWeightedMiner, HardestMiner, InBatchMiner, MultiSimilarityMiner, RandomMiner,
    SemiHardMiner,
};
pub use similarity::{cosine_similarity, dot, euclidean_distance, l2_normalize, pairwise_cosine};
