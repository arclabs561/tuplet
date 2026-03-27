pub mod losses;
pub mod mahalanobis;
#[cfg(feature = "mining")]
pub mod mining;
pub mod similarity;

pub use losses::{
    LossOutput, cosine_embedding_loss, infonce_loss, matryoshka_loss, mnrl_loss, triplet_loss,
};
pub use mahalanobis::{
    LmnnConfig, MahalanobisConfig, NcaConfig, learn_mahalanobis, learn_transform, lmnn,
    mahalanobis_distance, nca, transform, transform_batch,
};
#[cfg(feature = "mining")]
pub use mining::{HardestMiner, InBatchMiner, NegativeMiner, RandomMiner, SemiHardMiner};
pub use similarity::{cosine_similarity, dot, euclidean_distance, l2_normalize, pairwise_cosine};
