# ml-predict

Machine learning exercise, whose goal is to classify fetal health in order to prevent problems during pregnancy.

Data are taken from [Kaggle Fetal Health Classification dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

### Executing scripts
All python script is contained in source. The main executable is src/main.py.

#### src/main.py
main.py is written via _click_ library, and it is designed to run via CLI via commands like src/main.py --params args.
Run src/main.py --help to see all commands.

#### bin
bin folder contains main executables. They execute main script commands.

In order to run bin commands, you need to do the following:

- open your favorite terminal, and cd repo directory
- define a file called '.env' containing following variables:
  - WORKDIR=/path/to/working/directory  # where you cloned repository
  - DATAPATH=/path/to/data.csv  # where you saved original data
  - OUTPUTDIR=/path/to/output  # where you want to store output results
- Run executables from termial
The repository contains a test.env file emulating how the true .env should be.

Please note the following:
- All ./bin executables source .env, so you need to create it otherwise *they will not work*
- All ./bin executables assume a pipenv virtual environment has been created, so install it before
