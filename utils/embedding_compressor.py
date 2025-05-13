from sklearn.decomposition import PCA
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingCompressor:
    def __init__(self, target_dimensions: int = 2000):
        """Initialize the embedding compressor with target dimensions."""
        self.target_dimensions = target_dimensions
        self.pca = PCA(n_components=target_dimensions)
        self.is_fitted = False
        logger.info(f"Initialized embedding compressor with target dimensions: {target_dimensions}")

    def compress(self, embeddings: List[float]) -> List[float]:
        """Compress a single embedding vector to target dimensions."""
        try:
            # Reshape the embedding for PCA
            embedding_array = np.array(embeddings).reshape(1, -1)
            
            # Fit PCA if not already fitted
            if not self.is_fitted:
                logger.debug("Fitting PCA model for first use")
                self.pca.fit(embedding_array)
                self.is_fitted = True
            
            # Transform the embedding
            compressed = self.pca.transform(embedding_array)
            result = compressed.flatten().tolist()
            
            logger.debug(f"Successfully compressed embedding from {len(embeddings)} to {len(result)} dimensions")
            return result
        except Exception as e:
            logger.error(f"Error compressing embedding: {str(e)}", exc_info=True)
            raise

    def decompress(self, compressed_embedding: List[float]) -> List[float]:
        """Decompress a compressed embedding back to original dimensions (for comparison if needed)."""
        try:
            if not self.is_fitted:
                raise ValueError("Compressor must be fitted before decompression")
            
            compressed_array = np.array(compressed_embedding).reshape(1, -1)
            decompressed = self.pca.inverse_transform(compressed_array)
            result = decompressed.flatten().tolist()
            
            logger.debug(f"Successfully decompressed embedding from {len(compressed_embedding)} to {len(result)} dimensions")
            return result
        except Exception as e:
            logger.error(f"Error decompressing embedding: {str(e)}", exc_info=True)
            raise

# Create a singleton instance
compressor = EmbeddingCompressor() 