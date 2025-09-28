import dagster as dg
from . import assets
from . import resources


resources_defs = {
    "document_store_resource": resources.document_store_resource,
    "vector_store_resource": resources.vector_store_resource,
}

defs = dg.Definitions(
    assets=dg.load_assets_from_modules([assets]), resources=resources_defs
)
