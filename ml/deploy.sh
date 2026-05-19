#!/bin/bash
# BusyBee ML Pipeline - Deployment Script
# Deploys the ML prediction pipeline to AWS

set -e

# Configuration
ENVIRONMENT=${1:-production}
AWS_REGION=${AWS_REGION:-us-east-1}
STACK_NAME="busybee-ml-pipeline-${ENVIRONMENT}"
MODELS_DIR="./models"
LAMBDA_DIR="./lambda"

echo "=========================================="
echo "BusyBee ML Pipeline Deployment"
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${AWS_REGION}"
echo "=========================================="

# Check required environment variables
if [ -z "$SUPABASE_URL" ]; then
    echo "Error: SUPABASE_URL environment variable is required"
    exit 1
fi

if [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo "Error: SUPABASE_SERVICE_KEY environment variable is required"
    exit 1
fi

# Step 1: Create S3 bucket if it doesn't exist
BUCKET_NAME="busybee-ml-models-${ENVIRONMENT}"
echo "Step 1: Checking S3 bucket ${BUCKET_NAME}..."

if aws s3 ls "s3://${BUCKET_NAME}" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "Creating S3 bucket..."
    aws s3 mb "s3://${BUCKET_NAME}" --region ${AWS_REGION}
else
    echo "S3 bucket already exists"
fi

# Step 2: Build Lambda deployment package
echo "Step 2: Building Lambda deployment package..."
PACKAGE_DIR="./package"
rm -rf ${PACKAGE_DIR}
mkdir -p ${PACKAGE_DIR}

# Install dependencies
pip install -t ${PACKAGE_DIR} \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    supabase \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --python-version 3.11

# Copy Lambda handler
cp ${LAMBDA_DIR}/prediction_handler.py ${PACKAGE_DIR}/

# Create deployment zip
cd ${PACKAGE_DIR}
zip -r9 ../lambda-deployment.zip .
cd ..

echo "Lambda package created: lambda-deployment.zip"

# Step 3: Build Lambda Layer for ML dependencies
echo "Step 3: Building Lambda Layer..."
LAYER_DIR="./layer"
rm -rf ${LAYER_DIR}
mkdir -p ${LAYER_DIR}/python

pip install -t ${LAYER_DIR}/python \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    supabase \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --python-version 3.11

cd ${LAYER_DIR}
zip -r9 ../ml-dependencies.zip .
cd ..

# Upload layer to S3
echo "Uploading Lambda layer to S3..."
aws s3 cp ml-dependencies.zip "s3://${BUCKET_NAME}/layers/ml-dependencies.zip"

# Step 4: Upload trained models to S3
echo "Step 4: Uploading trained models..."
if [ -d "$MODELS_DIR" ] && [ "$(ls -A $MODELS_DIR)" ]; then
    aws s3 sync ${MODELS_DIR}/ "s3://${BUCKET_NAME}/models/"
    echo "Models uploaded successfully"
else
    echo "Warning: No models found in ${MODELS_DIR}. Run training first."
fi

# Step 5: Deploy CloudFormation stack
echo "Step 5: Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file ./aws/cloudformation-template.yaml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        Environment=${ENVIRONMENT} \
        SupabaseUrl=${SUPABASE_URL} \
        SupabaseServiceKey=${SUPABASE_SERVICE_KEY} \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${AWS_REGION}

# Step 6: Update Lambda function code
echo "Step 6: Updating Lambda function code..."
LAMBDA_FUNCTION_NAME="busybee-ml-predictions-${ENVIRONMENT}"
aws lambda update-function-code \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --zip-file fileb://lambda-deployment.zip \
    --region ${AWS_REGION}

# Step 7: Verify deployment
echo "Step 7: Verifying deployment..."
aws lambda get-function --function-name ${LAMBDA_FUNCTION_NAME} --region ${AWS_REGION}

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Train models: python training/train_models.py --use-sample-data --output-path ./models"
echo "2. Upload models: aws s3 sync ./models/ s3://${BUCKET_NAME}/models/"
echo "3. Test Lambda: aws lambda invoke --function-name ${LAMBDA_FUNCTION_NAME} --payload '{\"model_type\":\"all\",\"model_source\":\"s3\"}' response.json"
echo "4. Check logs: aws logs tail /aws/lambda/${LAMBDA_FUNCTION_NAME} --follow"
echo ""
echo "Predictions will run daily at 2 AM UTC"
