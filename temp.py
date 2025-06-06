from embedkit import EmbedKit
from embedkit.classes import Model, SnowflakeInputType

kit = EmbedKit.colpali(
    model=Model.ColPali.COLSMOL_256M,  # or COLSMOL_256M, COLSMOL_500M
)

result = kit.embed_text("Hello world")
print(result.objects[0].embedding.shape)

# Get image embeddings
result = kit.embed_image("tests/fixtures/2407.01449v6_p1.png")
print(result.objects[0].embedding.shape)
print(result.objects[0].source_b64[:10])
