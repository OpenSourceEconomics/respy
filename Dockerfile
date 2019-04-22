FROM continuumio/miniconda3:4.5.12

# Prepare conda environment.
COPY environment.yml .
RUN conda update conda -y
RUN conda init --all
RUN conda env create

# Install current version of respy.
COPY . respy/
RUN conda run --name respy pip install respy/

CMD ["/bin/bash"]
