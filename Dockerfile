# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
# ENV NAME World

# CMD mkdir /root/.ssh
# CMD echo "$POPPY_LOCAL ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBPolwwJWDyMxCJ+ibJWE/fAO0xQTC53sguEEtpryFAOTyBVf2BuszEJqVKMz0fG0MU5v1H+00ASK0FFGoenJWCM=" > /root/.ssh/known_hosts
# CMD cat /root/.ssh/known_hosts

# Run app.py when the container launches
# CMD ["python", "app.py"]


