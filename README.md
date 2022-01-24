# Project 4 - Udacity Nano Machine Learning 

### Data - [dog breed - Udacity](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

Project 4 - Image Classification on AWS. 

Tuning - Training - Deploy the endpoint on AWS

Lambda function is used for calling endpoint to predict image classification.

- Step 1: We download data to an S3 bucket and run our notebooks in a Sagemaker instance. 
- Step 2: We run our notebooks using EC2 ml.t3.medium conda_amazonei_pytorch_latest_p37 in our workspace. 
- Step 3: We use the Lambda function to revoke run time endpoint to predict. 
- Step 4: We use the IAM functions to set up security. 
- Step 5: We set up concurrency for our Lambda function and auto-scaling for our deployed endpoint in Sagemaker.

More details in writeup.pdf.



