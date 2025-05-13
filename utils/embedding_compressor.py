from typing import List
import numpy as np
import logging
from sklearn.random_projection import GaussianRandomProjection

logger = logging.getLogger(__name__)

class EmbeddingCompressor:
    def __init__(self, target_dimensions: int = 2000):
        """Initialize the embedding compressor with target dimensions."""
        self.target_dimensions = target_dimensions
        self.random_projection = GaussianRandomProjection(n_components=target_dimensions, random_state=42)
        self.projection_matrix = None
        logger.info(f"Initialized embedding compressor with target dimensions: {target_dimensions}")

    def compress(self, embeddings: List[float]) -> List[float]:
        """Compress a single embedding vector to target dimensions using random projection."""
        try:
            # Convert input to numpy array
            embedding_array = np.array(embeddings).reshape(1, -1)
            original_dim = embedding_array.shape[1]

            # Initialize projection matrix if not already done
            if self.projection_matrix is None:
                logger.debug("Initializing random projection matrix")
                self.random_projection.fit(np.zeros((1, original_dim)))
                self.projection_matrix = self.random_projection.components_

            # Apply projection
            compressed = np.dot(embedding_array, self.projection_matrix.T)
            result = compressed.flatten().tolist()
            
            logger.debug(f"Successfully compressed embedding from {len(embeddings)} to {len(result)} dimensions")
            return result
        except Exception as e:
            logger.error(f"Error compressing embedding: {str(e)}", exc_info=True)
            raise

    def decompress(self, compressed_embedding: List[float]) -> List[float]:
        """Approximate decompression using pseudo-inverse of projection matrix."""
        try:
            if self.projection_matrix is None:
                raise ValueError("Compressor must be initialized before decompression")
            
            compressed_array = np.array(compressed_embedding).reshape(1, -1)
            # Use pseudo-inverse for approximate reconstruction
            pseudo_inverse = np.linalg.pinv(self.projection_matrix)
            decompressed = np.dot(compressed_array, pseudo_inverse)
            result = decompressed.flatten().tolist()
            
            logger.debug(f"Successfully decompressed embedding from {len(compressed_embedding)} to {len(result)} dimensions")
            return result
        except Exception as e:
            logger.error(f"Error decompressing embedding: {str(e)}", exc_info=True)
            raise

# Create a singleton instance
compressor = EmbeddingCompressor() 