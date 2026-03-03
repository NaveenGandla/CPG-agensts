#!/usr/bin/env bash
# ============================================================
# Azure Container Apps deployment script
# Prerequisites: az CLI logged in, Docker installed
# ============================================================
set -euo pipefail

# ── Configuration ───────────────────────────────────────────
RESOURCE_GROUP="rg-cpg-agents"
LOCATION="eastus2"
ACR_NAME="acrcpgagents"
CONTAINER_APP_ENV="cae-cpg-agents"
CONTAINER_APP_NAME="ca-cpg-agents"
IMAGE_NAME="cpg-multi-agent"
IMAGE_TAG="latest"

# ── 1. Resource Group ──────────────────────────────────────
echo "▸ Creating resource group..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

# ── 2. Azure Container Registry ───────────────────────────
echo "▸ Creating container registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true

# ── 3. Build & push image ─────────────────────────────────
echo "▸ Building and pushing Docker image..."
az acr build \
  --registry "$ACR_NAME" \
  --image "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

# ── 4. Container Apps Environment ─────────────────────────
echo "▸ Creating Container Apps environment..."
az containerapp env create \
  --name "$CONTAINER_APP_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

# ── 5. Deploy Container App ──────────────────────────────
echo "▸ Deploying container app..."
az containerapp create \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_APP_ENV" \
  --image "${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --system-assigned \
  --env-vars \
    AZURE_OPENAI_ENDPOINT="<REPLACE>" \
    AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small" \
    AZURE_SEARCH_ENDPOINT="<REPLACE>" \
    APPLICATIONINSIGHTS_CONNECTION_STRING="<REPLACE>" \
    ENVIRONMENT="production"

# ── 6. Assign RBAC roles to Managed Identity ──────────────
echo "▸ Retrieving managed identity principal ID..."
PRINCIPAL_ID=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "identity.principalId" -o tsv)

echo "▸ Assigning RBAC roles..."

# Cognitive Services OpenAI User — for Azure OpenAI
az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Cognitive Services OpenAI User" \
  --scope "/subscriptions/<SUB_ID>/resourceGroups/<AOAI_RG>/providers/Microsoft.CognitiveServices/accounts/<AOAI_NAME>"

# Search Index Data Contributor — for Azure AI Search read/write
az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Search Index Data Contributor" \
  --scope "/subscriptions/<SUB_ID>/resourceGroups/<SEARCH_RG>/providers/Microsoft.Search/searchServices/<SEARCH_NAME>"

# Search Service Contributor — for index management
az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Search Service Contributor" \
  --scope "/subscriptions/<SUB_ID>/resourceGroups/<SEARCH_RG>/providers/Microsoft.Search/searchServices/<SEARCH_NAME>"

echo "▸ Deployment complete."
FQDN=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)
echo "  App URL: https://${FQDN}"
echo "  Health:  https://${FQDN}/api/v1/health"
