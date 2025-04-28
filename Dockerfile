# Use the official conda image as base
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app
RUN mkdir -p /app/model

# Copy environment.yml file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Copy the rest of the application
COPY . .

# Download the model
RUN conda run -n FBTR python download_model.py

# Expose port 8080
EXPOSE 8080

# Command to run the application using run.py
CMD ["conda", "run", "--no-capture-output", "-n", "FBTR", "python", "main.py"] 
