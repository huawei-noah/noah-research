SHELL := /bin/bash

PRESERVE_ENV_VARS := \
	MINDMEMOS_CONFIG_NAME \
	MINDMEMOS_CONFIG_PATH \
	MINDMEMOS_API_HOST \
	MINDMEMOS_API_PORT \
	MINDMEMOS_QDRANT_URL \
	MINDMEMOS_QDRANT_HTTP_PORT \
	MINDMEMOS_QDRANT_GRPC_PORT \
	MINDMEMOS_QDRANT_PREFER_GRPC \
	MINDMEMOS_QDRANT_API_KEY \
	MINDMEMOS_QDRANT_CPUS \
	MINDMEMOS_QDRANT_MEMORY_LIMIT \
	MINDMEMOS_QDRANT_MEMORY_RESERVATION \
	MINDMEMOS_NEO4J_URI \
	MINDMEMOS_NEO4J_HTTP_PORT \
	MINDMEMOS_NEO4J_BOLT_PORT \
	MINDMEMOS_NEO4J_USERNAME \
	MINDMEMOS_NEO4J_PASSWORD \
	MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS \
	MINDMEMOS_KAFKA_PORT \
	MINDMEMOS_KAFKA_UI_PORT \
	MINDMEMOS_KAFKA_EXPORTER_PORT \
	MINDMEMOS_KAFKA_NUM_PARTITIONS \
	MINDMEMOS_TELEMETRY_ENDPOINT \
	MINDMEMOS_OTEL_HTTP_PORT \
	MINDMEMOS_OTEL_GRPC_PORT \
	MINDMEMOS_OTEL_TTL \
	MINDMEMOS_GRAFANA_PORT \
	MINDMEMOS_GRAFANA_USER \
	MINDMEMOS_GRAFANA_PASSWORD \
	MINDMEMOS_CLICKHOUSE_HTTP_PORT \
	MINDMEMOS_CLICKHOUSE_NATIVE_PORT \
	MINDMEMOS_CLICKHOUSE_DB \
	MINDMEMOS_CLICKHOUSE_USER \
	MINDMEMOS_CLICKHOUSE_PASSWORD

define capture_external_env
ifneq ($(filter environment command line,$(origin $(1))),)
__external_$(1) := $($(1))
__external_set_$(1) := 1
endif
endef

$(foreach var,$(PRESERVE_ENV_VARS),$(eval $(call capture_external_env,$(var))))

-include .env

define restore_external_env
ifeq ($(__external_set_$(1)),1)
override $(1) := $(__external_$(1))
endif
endef

$(foreach var,$(PRESERVE_ENV_VARS),$(eval $(call restore_external_env,$(var))))

ifneq ($(__external_set_MINDMEMOS_CONFIG_NAME),)
ifeq ($(__external_set_MINDMEMOS_CONFIG_PATH),)
override MINDMEMOS_CONFIG_PATH :=
endif
endif

export

COMPOSE_ENV := $(if $(wildcard .env),--env-file .env,)
COMPOSE := docker compose $(COMPOSE_ENV) -f dockers/docker-compose.memory.yml
DEV_CORE_SERVICES := qdrant neo4j kafka kafka-ui kafka-exporter
DB_OBSERVABILITY_SERVICES := $(DEV_CORE_SERVICES) clickhouse otel-collector grafana
UV ?= $(shell command -v uv 2>/dev/null || printf uv)
MINDMEMOS_GRAFANA_PORT ?= $(or $(MINDMEM_GRAFANA_PORT),$(MEMOS_GRAFANA_PORT),3000)
MINDMEMOS_CLICKHOUSE_HTTP_PORT ?= $(or $(MINDMEM_CLICKHOUSE_HTTP_PORT),$(MEMOS_CLICKHOUSE_HTTP_PORT),8123)
MINDMEMOS_TELEMETRY_ENDPOINT ?= http://localhost:$(MINDMEMOS_OTEL_HTTP_PORT)
MINDMEMOS_OTEL_HTTP_PORT ?= $(or $(MINDMEM_OTEL_HTTP_PORT),$(MEMOS_OTEL_HTTP_PORT),4318)
MINDMEMOS_QDRANT_URL ?= http://localhost:$(MINDMEMOS_QDRANT_HTTP_PORT)
MINDMEMOS_QDRANT_HTTP_PORT ?= $(or $(MINDMEM_QDRANT_HTTP_PORT),$(MEMOS_QDRANT_HTTP_PORT),6333)
MINDMEMOS_QDRANT_GRPC_PORT ?= $(or $(MINDMEM_QDRANT_GRPC_PORT),$(MEMOS_QDRANT_GRPC_PORT),6334)
MINDMEMOS_NEO4J_URI ?= bolt://localhost:$(MINDMEMOS_NEO4J_BOLT_PORT)
MINDMEMOS_NEO4J_HTTP_PORT ?= $(or $(MINDMEM_NEO4J_HTTP_PORT),$(MEMOS_NEO4J_HTTP_PORT),7474)
MINDMEMOS_NEO4J_BOLT_PORT ?= $(or $(MINDMEM_NEO4J_BOLT_PORT),$(MEMOS_NEO4J_BOLT_PORT),7687)
MINDMEMOS_NEO4J_USERNAME ?= $(or $(MINDMEM_NEO4J_USERNAME),$(MEMOS_NEO4J_USERNAME),neo4j)
MINDMEMOS_NEO4J_PASSWORD ?= $(or $(MINDMEM_NEO4J_PASSWORD),$(MEMOS_NEO4J_PASSWORD),mindmemos_dev_password)
MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS ?= localhost:$(MINDMEMOS_KAFKA_PORT)
MINDMEMOS_KAFKA_PORT ?= $(or $(MINDMEM_KAFKA_PORT),$(MEMOS_KAFKA_PORT),9092)
MINDMEMOS_KAFKA_UI_PORT ?= $(or $(MINDMEM_KAFKA_UI_PORT),$(MEMOS_KAFKA_UI_PORT),8080)
API_HOST ?= $(or $(MINDMEMOS_API_HOST),127.0.0.1)
API_PORT ?= $(or $(MINDMEMOS_API_PORT),8000)
API_RELOAD ?= 0
API_RELOAD_FLAGS := $(if $(filter 1 true yes,$(API_RELOAD)),--reload,)
PROFILE_OUT ?= tmp/profile.svg
PROFILE_SECONDS ?= 60
PROFILE_RATE ?= 100
PROFILE_PID ?= $(PID)

.PHONY: dev dev-setup hooks-install format lint dev-core api profile-api db db-observability db-clean dev-down

dev-setup:
	$(UV) sync
	$(UV) run pre-commit install --install-hooks
	$(UV) run python scripts/install_nlp_assets.py
	$(UV) run python scripts/check_nlp_assets.py

hooks-install:
	$(UV) run pre-commit install --install-hooks

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

lint:
	$(UV) run ruff check .

dev: db-observability
	@echo "Use 'make dev-down' to stop databases."
	@echo "Starting FastAPI at http://$(API_HOST):$(API_PORT) (Ctrl-C to stop)"
	$(UV) run uvicorn mindmemos.api.app:app --host $(API_HOST) --port $(API_PORT) $(API_RELOAD_FLAGS)

api:
	$(UV) run uvicorn mindmemos.api.app:app --host $(API_HOST) --port $(API_PORT) $(API_RELOAD_FLAGS)

profile-api:
	@pid="$(PROFILE_PID)"; \
	if [ -z "$$pid" ]; then \
		pid="$$(lsof -tiTCP:$(API_PORT) -sTCP:LISTEN 2>/dev/null | head -n 1)"; \
	fi; \
	if [ -z "$$pid" ]; then \
		echo "No FastAPI process found on port $(API_PORT). Run 'make dev' or pass PID=<pid>."; \
		exit 1; \
	fi; \
	mkdir -p "$$(dirname "$(PROFILE_OUT)")"; \
	echo "Recording CPU profile from PID $$pid for $(PROFILE_SECONDS)s -> $(PROFILE_OUT)"; \
	$(UV) run py-spy record --pid "$$pid" --subprocesses --duration "$(PROFILE_SECONDS)" --rate "$(PROFILE_RATE)" --output "$(PROFILE_OUT)"

db: db-observability

dev-core:
	$(COMPOSE) up -d --wait $(DEV_CORE_SERVICES)
	@echo "Kafka:      $(MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS)"
	@echo "Kafka UI:   http://localhost:$(MINDMEMOS_KAFKA_UI_PORT)"
	@echo "Qdrant:     $(MINDMEMOS_QDRANT_URL)"
	@echo "Neo4j:      $(MINDMEMOS_NEO4J_URI)"

db-observability:
	$(COMPOSE) up -d --wait $(DB_OBSERVABILITY_SERVICES)
	@echo "Grafana:    http://localhost:$(MINDMEMOS_GRAFANA_PORT) (dashboard: MindMemOS Observability)"
	@echo "ClickHouse: http://localhost:$(MINDMEMOS_CLICKHOUSE_HTTP_PORT)"
	@echo "Kafka:      $(MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS)"
	@echo "Qdrant:     $(MINDMEMOS_QDRANT_URL)"
	@echo "Neo4j:      $(MINDMEMOS_NEO4J_URI)"
	@echo "OTel OTLP:  $(MINDMEMOS_TELEMETRY_ENDPOINT)"

db-clean:
	$(COMPOSE) down -v

dev-down:
	$(COMPOSE) down
