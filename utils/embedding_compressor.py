from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingCompressor:
    def __init__(self, target_dimensions: int = 1536):
        """Initialize the embedding compressor with target dimensions."""
        self.target_dimensions = target_dimensions
        logger.info(f"Initialized embedding compressor with target dimensions: {target_dimensions}")

    def compress(self, embeddings: List[float]) -> List[float]:
        """Pass through the embeddings since we're using native Azure OpenAI dimensions."""
        try:
            if len(embeddings) != self.target_dimensions:
                raise ValueError(f"Expected embedding dimension {self.target_dimensions}, got {len(embeddings)}")
            return embeddings
        except Exception as e:
            logger.error(f"Error processing embedding: {str(e)}", exc_info=True)
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