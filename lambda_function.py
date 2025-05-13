import json
import boto3
import os
from datetime import datetime
import uuid
from PIL import Image
import base64
from openai import OpenAI

#Intiialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ConversationHistory')

#intialize S3 client
s3 = boto3.client('s3')

#intialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def validate_and_explain_math_problem(problem, context=None):
    """
    Process a math problem with optional conversation context.
    
    If context is provided, this is treated as a follow-up question to a previous problem.
    The context includes the original problem (text and possibly image) and the previous response.
    """
    # Validate the problem
    if not problem:
        return {
            'statusCode': 400,
            'body': json.dumps('No problem provided')
        }
    
def validate_and_explain_math_problem_image(image_data):
    """
    Validate the image data and explain the math problem.
    """
    # Validate the image data
    if not is_valid_base64(image_data):
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid image data')
        }
    
    # Process the image data
    try:
        # Process the image data
        image_data = base64.b64decode(image_data)

        # Upload the image to S3
        s3.put_object(Bucket=os.getenv('S3_BUCKET_NAME'), Key=s3_key, Body=image_data) 
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing image: {str(e)}')
        }
    
    

def is_valid_base64(s):
    try:
        # Check if string can be decoded as base64
        base64.b64decode(s)
        return True
    except Exception:
        return False
        
def generate_user_id():
    return str(uuid.uuid4())

def get_conversation_history(user_id):
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id),
    )
    return response.get('Items', [])
def log_conversation(user_id, problem, explanation, image_url=None):
    timestamp = datetime.now().isoformat()
    item = {
        'user_id': user_id,
        'timestamp': timestamp,
        'problem': problem,
        'explanation': explanation,
    }
    if image_url:
        item['image_data'] = image_url
    table.put_item(Item=item)

def lambda_function_handler(event, context):
    print(f"Event: {event}")
    try:
        body = event.get("body", None)
        if body is None:
            body = event

        print(f"Raw body: {body}")

        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception as e:
                print(f"Error parsing request body JSON: {e}")
                return {
                    "statusCode": 400,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type"
                    },
                    "body": json.dumps({"error": "Invalid JSON in request body"})
                }
            
        print(f"Processed body: {body}")

        user_id = body.get("user_id")
        is_new_user = False
        if not user_id:
            user_id = generate_user_id()
            is_new_user = True
            print(f"Generated new user ID: {user_id}")

        
        input_type = body.get("input_type", "")
        problem = body.get("problem", "")
        image_data = body.get("image_data", None)

        # Get the conversation context - check both 'context' and fallback to 'conversationContext'
        follow_up_context =  body.get("context", body.get("conversationContext"))

        # If we have user_id by no context, try to get the last conversation for context
        if user_id and not follow_up_context and body.get("isFollowUp", False):
            history = get_conversation_history(user_id)
            if history:
                sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
                if sorted_history:
                    last_conversation = sorted_history[0]
                    follow_up_context = {
                        'isFollowUp': True,
                        'originalProblem': {
                            'text': last_conversation.get('problem', ''),
                            'image': last_conversation.get('image_data', '')
                        },
                        'previousResponse': last_conversation.get('explanation', '')
                    }
                    print(f"Created follow-up context from history: {follow_up_context}")

        print(f"Input type: {input_type}")
        print(f"Problem: '{problem[:50]}...'" if len(problem) > 50 else f"Problem: '{problem}'")
        print(f"Has image data: {image_data is not None}")
        print(f"Has follow-up context: {follow_up_context is not None}")

        # Hnalde follow-up questions with context
        if follow_up_context:
            print(f"Follow-up context: {follow_up_context}")

            # Process follow-up questions for both text and image inputs
            if (input_type == "text" and problem) or (input_type == "image" and image_data):
                try:
                    print("Processing follow-up question...")
                    # Ensure isFollowUp is set to True
                    if not follow_up_context.get('isFollowUp'):
                        follow_up_context['isFollowUp'] = True

                    # for text input type
                    if input_type == "text":
                        explanation = validate_and_explain_math_problem(problem, follow_up_context)
                        log_conversation(user_id, problem, explanation) #Might need to be updated
                        response_body = {
                            "problem": problem,
                            "explanation": explanation,
                            "user_id": user_id,
                        }
                    elif input_type == "image":
                        #first store the image in the context for future reference
                        if not follow_up_context.get('originalProblem'):
                            follow_up_context['originalProblem'] = {}

                        # if there's new image data, use it tas the follow-up question
                        if is_valid_base64(image_data):
                            # use the validate_and_explain_math_problem to process the follow-up with image context
                            # THe follow-up is the new image, but we';; use the same text context
                            follow_up_context['originalProblem']['image'] = image_data
                            # use empty text as the follow-up question since the image is the question
                            explanation = validate_and_explain_math_problem("", follow_up_context)
                            log_conversation(user_id, image_data, explanation)
                            response_body = {
                                "problem": image_data,
                                "explanation": explanation,
                                "user_id": user_id,
                            }
                        else:
                            response_body = {
                                "explanation": "No valid image data provided",
                                "user_id": user_id,
                            }
                    return {
                        "statusCode": 200,
                        "headers": {
                            "Content-Type": "application/json",
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Headers": "Content-Type"
                        },
                        "body": json.dumps(response_body)
                    }
                except Exception as e:
                    print(f"Error processing follow-up question: {e}")
                    return {
                        "statusCode": 500,
                        "headers": {
                            "Content-Type": "application/json",
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Headers": "Content-Type"
                        },
                        "body": json.dumps({"error": f"Error processing follow-up: {str(e)}"})
                    }
                
            #hnald normal flow for new questins
            
            if input_type == "text" and image_data:
                #validate base64 image data
                if not is_valid_base64(image_data):
                    response_body = {
                        "explanation": "Please provide a valid image.",
                    }
                else:
                    try:
                        # process image with GPT to extract mathematical problem
                        explanation = validate_and_explain_math_problem_image(image_data)
                        log_conversation(user_id, image_data, explanation, image_data)
                        response_body = {
                            "problem": image_data,
                            "explanation": explanation,
                            "user_id": user_id,
                            "requires_confirmation": False,
                            "original_image": image_data  # Include the original image in the response
                        }
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        response_body = {
                            "explanation": f"Error processing image: {str(e)}",
                            "user_id": user_id
                        }
            elif input_type == "text":
                # Process text problem (whether it's a new input or a confirmation)
                if problem:
                    try:
                        # pass none for context since this is not a follow-up
                        explanation = validate_and_explain_math_problem(problem, None)
                        log_conversation(user_id, problem, explanation)
                        response_body = {
                            "problem": problem,
                            "explanation": explanation,
                            "user_id": user_id,
                        }
                    except Exception as e:
                        print(f"Error processing text problem: {e}")
                        response_body = {
                            "explanation": f"Error processing text problem: {str(e)}",
                            "user_id": user_id
                        }
                else:
                    response_body = {
                        "explanation": "Please provide a valid math problem.",
                        "user_id": user_id
                    }
            else:
                response_body = {
                    "explanation": "Please provide a valid input type (text or image).",
                    "user_id": user_id
                }

            response_body["is_new_user"] = is_new_user

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type"
                },
                "body": json.dumps(response_body)
            }
    except Exception as e:
        print(f"Error in lambda_function_handler: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"error": str(e)})
        }
