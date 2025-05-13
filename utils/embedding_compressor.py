from typing import List
import numpy as np
import logging
from sklearn.random_projection import GaussianRandomProjection
from config.config import config

logger = logging.getLogger(__name__)

class EmbeddingCompressor:
    def __init__(self):
        """Initialize the embedding compressor with source and target dimensions."""
        self.source_dimensions = config.source_embedding_dimension
        self.target_dimensions = config.target_embedding_dimension
        self.random_projection = GaussianRandomProjection(
            n_components=self.target_dimensions,
            random_state=42  # Fixed seed for consistency
        )
        self.projection_matrix = None
        logger.info(
            f"Initialized embedding compressor: {self.source_dimensions}D → {self.target_dimensions}D"
        )

    def compress(self, embeddings: List[float]) -> List[float]:
        """Compress embeddings from source to target dimensions using random projection."""
        try:
            # If already at target dimensions, return as is
            if len(embeddings) == self.target_dimensions:
                logger.debug(f"Embedding already at target dimensions ({len(embeddings)}D)")
                return embeddings
            
            # Validate input dimensions
            if len(embeddings) != self.source_dimensions:
                raise ValueError(
                    f"Expected {self.source_dimensions} dimensions, got {len(embeddings)}"
                )
            
            # Convert to numpy array and reshape
            embedding_array = np.array(embeddings).reshape(1, -1)
            
            # Initialize projection matrix if not already done
            if self.projection_matrix is None:
                logger.debug("Initializing random projection matrix")
                self.random_projection.fit(np.zeros((1, self.source_dimensions)))
                self.projection_matrix = self.random_projection.components_
            
            # Apply projection
            compressed = np.dot(embedding_array, self.projection_matrix.T)
            result = compressed.flatten().tolist()
            
            logger.debug(
                f"Successfully compressed embedding: {self.source_dimensions}D → {self.target_dimensions}D"
            )
            return result
            
        except Exception as e:
            logger.error(f"Error compressing embedding: {str(e)}", exc_info=True)
            raise

    def decompress(self, compressed_embedding: List[float]) -> List[float]:
        """Approximate decompression using pseudo-inverse of projection matrix."""
        try:
            # If at source dimensions, return as is
            if len(compressed_embedding) == self.source_dimensions:
                logger.debug(f"Embedding already at source dimensions ({len(compressed_embedding)}D)")
                return compressed_embedding
            
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