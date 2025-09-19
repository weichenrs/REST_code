import os
try:
    world_size = int(os.environ['WORLD_SIZE'])
except KeyError as e:
    raise RuntimeError(
        f"Could not find {e} in the torch environment, visit https://www.colossalai.org/ for more information on launching with torch"
    )

parallel = dict(
    data=1,
    pipeline=1,
    tensor=dict(size=world_size, mode='sequence')
)
