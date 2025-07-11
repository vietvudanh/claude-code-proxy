import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Helper function to clean schema for Gemini


def clean_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(
                    f"Removing unsupported format '{schema['format']}' for string type in Gemini schema."
                )
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()):  # Use list() to allow modification
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema
