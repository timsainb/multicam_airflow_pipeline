# Multicamera Analysis pipeline with Airflow automation

This library is comprised of a series of analysis tasks for working with video data collected with our library [multicamera_acquisition](https://github.com/dattalab-6-cam/multicamera_acquisition). 

The first version of the pipeline is stored in multicamera_airflow_pipeline.tim_240731. It comprises the following steps:
- 2D keypoint prediction
- Camera sync
- 3D triangulation
- Gimbal refinement
- Size normalization
- Arena alignment (align 3d keypoints to rectangular arena)
- Egocentric alignment to mouse
- Continuous variable computation
- Sync ephys to camera frames
- Spikesorting (kilosort4) ephys signals

Because these steps comprise a complex dependency graph, we use Apache Airflow to manage the pipeline. Because these tasks are too computationally expensive to run on a single computer, all computations are run on the HMS data cluster O2. 

### How it works
1. Airflow is installed on a local computer. 
2. /n/groups/datta is mounted on the local computer via `sshfs`
3. Airflow reads which recordings to process from a [google sheet](https://docs.google.com/spreadsheets/d/1jACsUmxuJ9Une59qmvzZGc1qXezKhKzD1zho2sEfcrU/edit?gid=0#gid=0)
4. For each recording, Airflow generates a DAG with all of the jobs it needs to run. 
5. Jobs are submitted to o2 using `sbatch` after waiting until their dependencies are completed. 

### Creating a new pipeline
[TODO]: You can make a new folder alongside tim_240731.

### Setting up your own airflow
[TODO] After installing airflow on your own computer. 

- Switch airflow config setting from `SequentialExecutor` (`executor = LocalExecutor` in `airflow.cfg`) for parallelization
- Switch to `postgresql` from `SQLite`

```
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

setup the database
```
sudo -i -u postgres
psql
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_pass';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;
-- PostgreSQL 15 requires additional privileges:
GRANT ALL ON SCHEMA public TO airflow_user;
\q
```
Then reload
```
sudo systemctl reload postgresql
```

update the config from
```
#sql_alchemy_conn = sqlite:////home/peromoseq/airflow/airflow.db
```
to 
```
sql_alchemy_conn = postgresql+psycopg2://airflow_user:airflow_pass@localhost/airflow_db
```
Then create a user and password to use
```
airflow users create \
    --role Admin \
    --username timsainb \
    --password <password> \
    --firstname tim \
    --lastname sainburg \
    --email timsainb@gmail.com
```

##### Starting Airflow
- To run airflow, in one terminal tab type: `airflow webserver`
    - alternatively, use tmux to run these in the background
- In a second tab, type: `airflow scheduler`
- Then navigate to `http://localhost:8080/` in your browser. 

##### Adding your Airflow job
Airflow DAGs are generated dynamically when new rows are added to your google sheet. 

If you want to run your own airflow, you can add `mydag.py` to `~/airflow/dags`, where `mydag.py` contains

```{python}
from multicamera_airflow_pipeline.tim_240731.airflow.dag import AirflowDAG

# gets the dag
airflow_dag = AirflowDAG(
    spreadsheet_url="https://docs.google.com/spreadsheet/ccc?key=1jACsUmxuJ9Une59qmvzZGc1qXezKhKzD1zho2sEfcrU&output=csv&gid=0",
)

# runs the dag
airflow_dag.run()

```

### TODO all
- estimate unit locations (?)