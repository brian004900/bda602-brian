FROM python:3.8

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Copy over code
COPY sql.sh /
COPY baseball.sql .

RUN chmod +x /sql.sh

ENTRYPOINT ["/sql.sh"]
