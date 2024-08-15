from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.models import Pool
from airflow.utils.dates import days_ago
from airflow.settings import Session


# Function to create pool if it doesn't exist
def create_pool_if_not_exists(pool_name, slots, description, include_deferred):
    session = Session()
    pool_exists = session.query(Pool).filter(Pool.pool == pool_name).first()
    if not pool_exists:
        new_pool = Pool(
            pool=pool_name,
            slots=slots,
            description=description,
            include_deferred=include_deferred,
        )
        session.add(new_pool)
        session.commit()
    else:
        # If pool exists, ensure it has the correct number of slots
        if pool_exists.slots != slots:
            pool_exists.slots = slots
            session.commit()
    session.close()


max_gpu_tasks = 3
include_deferred = False  # Or you can set it based on your business logic
# Create the pool with a maximum of 3 slots
create_pool_if_not_exists(
    "local_gpu_pool",
    max_gpu_tasks,
    "This is a local GPU pool for pero, with 3 possible jobs (corresponding to GPUs)",
    include_deferred,
)
