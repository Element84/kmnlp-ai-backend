ARG UV_VERSION=0.7.2
ARG PYTHON_VERSION=3.12

######################
# Create build image #
######################

# Start from UV image
# Do not use slim version because we at least want git installed for dependency resolution
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm AS build

WORKDIR /opt/src/

# Install project dependencies
COPY ./pyproject.toml .
COPY ./uv.lock .

# Copy src
COPY ./ /opt/src/

# Create venv and install dependencies
RUN uv lock --check
RUN uv sync --all-extras

######################
# Create final image #
######################

# Start from slim image. Needs python because uv python is a link.
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm-slim

# Copy files from build image
COPY --from=build /opt/src /opt/src

# Add uv to path
ENV PATH="/opt/src/.venv/bin:$PATH"

# Add libexpat1 for rasterio
RUN apt-get update && apt-get install libexpat1 && apt-get clean

WORKDIR /opt/src

CMD ["bash"]
