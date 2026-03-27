pub mod losses;
pub mod mahalanobis;
#[cfg(feature = "mining")]
pub mod mining;
pub mod similarity;

pub use losses::{
    LossOutput, circle_loss, contrastive_loss, cosine_embedding_loss, infonce_loss,
    lifted_structured_loss, matryoshka_loss, mnrl_loss, multi_similarity_loss, n_pairs_loss,
    supcon_loss, triplet_loss,
};
pub use mahalanobis::{
    ItmlConfig, LmnnConfig, MahalanobisConfig, NcaConfig, itml, kissme, learn_mahalanobis,
    learn_transform, lmnn, mahalanobis_distance, nca, transform, transform_batch,
};
#[cfg(feature = "mining")]
pub use mining::{
    DistanceWeightedMiner, HardestMiner, InBatchMiner, MultiSimilarityMiner, NegativeMiner,
    RandomMiner, SemiHardMiner,
};
pub use similarity::{cosine_similarity, dot, euclidean_distance, l2_normalize, pairwise_cosine};
