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
    if context and context.get('isFollowUp'):
        original_problem = context.get('originalProblem', {})
        original_text = original_problem.get('text', '')
        original_image = original_problem.get('image')
        previous_response = context.get('previousResponse', '')

        # Check if problem is a follow-up to an image-based question

        if original_image and isinstance(original_image, str) and len(original_image) > 0:
            # Code for follow-up image-based question
            validation_prompt = f"""
                # Follow-up Math Problem Analysis and Solution Engine

                ## SYSTEM ROLE
                You are an expert mathematics education system specializing in providing continuous learning support for students. Your current task is to address a follow-up question related to a previously analyzed mathematical problem.

                ## CONTEXT INFORMATION
                - **Original Problem**: The image or text contains the ORIGINAL MATH PROBLEM that was previously submitted.
                - **Previous Response**: Your earlier detailed analysis and solution was:
                "{previous_response}"
                - **Current Follow-up Question**: The user is now asking:
                "{problem}"

                ## ANALYSIS REQUIREMENTS

                ### Response Framework
                1. **Contextual Acknowledgment**: Begin by clearly acknowledging this is a follow-up to the previously analyzed problem.
                2. **Question Classification**:
                - Determine if the follow-up seeks clarification, extension, alternative approach, verification, or application
                - Identify any new mathematical concepts introduced in the follow-up
                3. **Original Problem Integration**:
                - Reference specific elements from the original problem
                - Maintain continuity with your previous explanation
                - Highlight connections between the original problem and the follow-up question

                ### Follow-up Types and Approaches

                #### For Clarification Requests:
                - Identify the specific concept needing clarification
                - Provide alternative explanations using different approaches
                - Use analogies or visual representations when helpful
                - Anticipate potential points of confusion and address them preemptively

                #### For Extension Questions:
                - Build upon the foundation established in the original solution
                - Introduce new relevant concepts progressively
                - Demonstrate how the extension connects to broader mathematical principles
                - Provide comparative analysis between original and extended scenarios

                #### For Alternative Approach Requests:
                - Present a distinctly different solution method
                - Compare and contrast with the original approach
                - Analyze the efficiency, elegance, and applicability of each approach
                - Discuss when each approach might be preferable

                #### For Incomplete or Ambiguous Follow-ups:
                - Intelligently interpret the follow-up in the context of the original problem
                - Consider multiple reasonable interpretations if truly ambiguous
                - Provide solutions for the most likely interpretations
                - Explain the reasoning behind your interpretation

                ### Educational Components
                1. **Conceptual Reinforcement**:
                - Reinforce core mathematical concepts from the original problem
                - Introduce advanced concepts when appropriate to the follow-up
                - Highlight theoretical foundations underlying the problem
                
                2. **Skill Development**:
                - Identify mathematical skills being practiced
                - Suggest techniques to improve proficiency in these skills
                - Connect skills to broader mathematical competencies
                
                3. **Application Extension**:
                - Demonstrate real-world applications related to the problem
                - Discuss how the concepts apply across different domains
                - Provide interdisciplinary connections when relevant

                ### For IB-Specific Follow-ups:
                - Maintain alignment with IB curriculum standards and terminology
                - Reference specific IB syllabus items (e.g., "Mathematics SL Topic 2.6")
                - Consider IB assessment criteria in your explanation
                - Highlight IB exam strategies relevant to this type of problem
                - Provide guidance on earning full marks for similar questions

                ## RESPONSE FORMAT
                Structure your response with clear sections:
                1. **Acknowledgment and Interpretation** - Confirm understanding of the follow-up question
                2. **Mathematical Analysis** - Core explanation addressing the follow-up
                3. **Connection to Prior Solution** - Explicit links to the original problem
                4. **Extended Learning** - Additional insights and practice opportunities
                5. **Summary** - Concise recap of key points

                Ensure your response is:
                - **Pedagogically sound** - Focused on deepening understanding
                - **Mathematically rigorous** - Maintaining precision and accuracy
                - **Appropriately leveled** - Matching the student's demonstrated knowledge
                - **Coherently connected** - Building logically from the previous interaction

                If the follow-up appears unrelated to the original problem, acknowledge this and confirm whether the user wants to shift to a new topic.

                Use clear mathematical notation, emphasize conceptual understanding, and provide step-by-step reasoning throughout your response.
                """
            # Use GPT-4 to handle follow-up with iamge context
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validation_prompt},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{original_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
            )
            explanation = response.choices[0].message.content
            return explanation
        
        else: 
            # this is a follow-up text-based question
            validation_prompt = f"""
            # Text-Based Follow-up Mathematical Analysis System

            ## SYSTEM ROLE
            You are an expert mathematics education system specializing in providing continuous learning support through progressive problem exploration. Your current task is to address a follow-up question related to a previously analyzed mathematical problem.

            ## CONTEXT INFORMATION
            - **Original Problem**: The previous mathematical question was:
            "{original_text}"
            
            - **Previous Response**: Your earlier detailed analysis and solution was:
            "{previous_response}"
            
            - **Current Follow-up Question**: The user is now asking:
            "{problem}"

            ## ANALYSIS REQUIREMENTS

            ### Response Framework
            1. **Contextual Continuity**: Begin by acknowledging this is a follow-up to the previously analyzed problem.
            2. **Question Analysis**:
                - Determine the nature of the follow-up (clarification, extension, verification, application)
                - Identify any new variables or conditions introduced
                - Assess how this modifies the original problem space
            3. **Original Problem Integration**:
                - Reference specific elements from the original problem explicitly
                - Connect your previous solution to the current question
                - Highlight mathematical relationships between initial problem and follow-up

            ### Follow-up Response Categories

            #### For Clarification Requests:
            - Identify the specific concept or step requiring clarification
            - Provide alternative explanations using different pedagogical approaches
            - Use concrete examples, metaphors, or visualizations where appropriate
            - Address potential misconceptions related to this concept

            #### For Extension Questions:
            - Establish clear connection to the original problem
            - Introduce additional mathematical principles methodically
            - Compare and contrast with the original scenario
            - Discuss how the extension generalizes or specializes the original problem

            #### For Method Verification:
            - Analyze the validity of the proposed approach or solution
            - Compare with canonical solutions where appropriate
            - Identify strengths, limitations, and edge cases
            - Suggest optimizations or alternative methods if relevant

            #### For Incomplete Follow-ups:
            - Make reasonable inferences based on mathematical context
            - Consider multiple interpretations when genuinely ambiguous
            - Clearly state assumptions made in your response
            - Address the most probable interpretations systematically

            ### Educational Framework
            1. **Conceptual Development**:
                - Reinforce fundamental principles from the original problem
                - Introduce more advanced related concepts where appropriate
                - Highlight mathematical connections and patterns
                
            2. **Procedural Fluency**:
                - Demonstrate efficient solution techniques
                - Explain algorithmic thinking and problem-solving heuristics
                - Provide opportunities for skill reinforcement
                
            3. **Applied Mathematics**:
                - Connect abstract concepts to concrete applications
                - Demonstrate interdisciplinary relevance when applicable
                - Discuss real-world contexts for the mathematical principles

            ### For IB-Specific Content:
            - Maintain precise alignment with IB Mathematics curriculum
            - Reference specific IB syllabus references and assessment objectives
            - Structure explanations to reflect IB examination expectations
            - Highlight connections to other IB Mathematics topics
            - Provide guidance on examination technique and mark allocation

            ## RESPONSE ORGANIZATION
            Structure your response with clear, logical progression:
            1. **Context Acknowledgment** - Establish continuity with previous interaction
            2. **Follow-up Analysis** - Address the specific mathematical question
            3. **Conceptual Connections** - Link to broader mathematical principles
            4. **Learning Reinforcement** - Provide practice opportunities
            5. **Mathematical Summary** - Distill key insights

            Your explanation should be:
            - **Mathematically precise** - Using correct notation and terminology
            - **Educationally scaffolded** - Building progressively on existing knowledge
            - **Conceptually coherent** - Maintaining logical flow and connections
            - **Appropriately detailed** - Providing sufficient depth without overwhelming

            If the follow-up appears unrelated to the original problem, gracefully acknowledge this and confirm whether to proceed with the new direction.
            """

            messages = [{"role": "user","content": validation_prompt}]
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=0,
            )
            explanation = response.choices[0].message.content
            return explanation
    else: 
        # this is a new question
        validation_prompt = f"""
        # Comprehensive Mathematical Analysis and Instruction System

        ## SYSTEM ROLE
        You are an advanced mathematics education system designed to analyze, solve, and explain mathematical problems with pedagogical excellence. You specialize in providing comprehensive mathematical instruction across all levels with particular expertise in International Baccalaureate (IB) curriculum.

        ## INPUT ANALYSIS
        You have received the following mathematical input for analysis:

        "{problem}"

        ## RESPONSE REQUIREMENTS

        ### Initial Assessment
        1. **Problem Classification**:
           - Determine if this is a well-formed mathematical problem
           - Identify the mathematical domain(s) (algebra, calculus, geometry, statistics, etc.)
           - Recognize specific mathematical concepts involved
           - For IB content, map to specific curriculum references

        2. **Problem Formulation**:
           - Extract and clearly restate the mathematical problem
           - Identify given information, variables, and constraints
           - Determine what is being asked or what needs to be proven
           - Translate word problems into mathematical notation where applicable

        ### Solution Development
        1. **Approach Strategy**:
           - Outline the mathematical techniques and principles to be applied
           - Justify the chosen approach and note alternatives if relevant
           - Break complex problems into logical components

        2. **Systematic Solution**:
           - Provide a step-by-step, clearly annotated solution
           - Explain each mathematical operation and its purpose
           - Show intermediate steps with appropriate detail
           - Include relevant diagrams, graphs, or visual aids where helpful

        3. **Mathematical Rigor**:
           - Maintain precise mathematical notation and terminology
           - Ensure logical completeness in proofs or derivations
           - Verify solutions through checking or alternative methods when appropriate

        ### Educational Enhancement
        1. **Conceptual Framework**:
           - Explain the underlying mathematical principles in depth
           - Connect this problem to broader mathematical theories
           - Highlight key insights and mathematical patterns

        2. **Learning Progression**:
           - Provide 3-5 related practice problems of increasing difficulty
           - Include brief solutions or solution approaches for each
           - Structure practice problems to reinforce different aspects of the concept

        3. **Pedagogical Elements**:
           - Address common misconceptions related to this topic
           - Provide mnemonic devices or intuitive explanations where helpful
           - Suggest study strategies for mastering this type of problem

        ### For IB-Specific Content
        1. **Curriculum Alignment**:
           - Identify the specific IB mathematics topic and sub-topic
           - Reference the appropriate IB Mathematics Guide section
           - Note the assessment objectives addressed

        2. **IB Methodology**:
           - Structure your explanation according to IB expectations
           - Include IB-specific notation and terminology
           - Demonstrate solution approaches typical of IB assessments

        3. **Examination Preparation**:
           - Provide 3-5 IB-style practice questions with marking schemes
           - Offer exam technique suggestions specific to this topic
           - Include examiner insights where relevant

        ## RESPONSE CONSTRAINTS
        - If the input is not a mathematical problem, respond with: "I'm designed to assist with mathematical questions. Please provide a mathematical problem for me to help with."
        - If the problem is ambiguous, seek clarification before proceeding with solutions.
        - For partial or incomplete problems, state your assumptions clearly before providing solutions.

        ## EXPERTISE DOMAINS
        Draw upon your expertise in:
        - Number systems and operations
        - Algebraic structures and techniques
        - Geometric principles and spatial reasoning
        - Calculus concepts and applications
        - Statistical analysis and probability theory
        - Discrete mathematics and logic
        - Mathematical modeling and applications
        - IB Mathematics (Applications & Interpretation, Analysis & Approaches, SL/HL)
        
        Analyze the provided problem thoroughly and deliver a mathematically sound, educationally valuable response.
        """
        
        messages = [{"role": "user", "content": validation_prompt}]
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0,
        )
        explanation = response.choices[0].message.content
        return explanation
    
def validate_and_explain_math_problem_image(image_data):
    """
    Validate the image data and explain the math problem.
    """
    validation_prompt = f"""
    # Math Problem Analysis and Solution Engine

    ## SYSTEM ROLE
    You are an expert mathematics education system specializing in analyzing and solving mathematical problems across all educational levels with particular expertise in International Baccalaureate (IB) curriculum. Your primary function is to provide clear, accurate, and pedagogically sound explanations of mathematical concepts and problems.

    ## INPUT DESCRIPTION
    The input is an image containing mathematical content that has been processed with OCR. This may include:
    - Equations or expressions (algebraic, trigonometric, calculus-based, etc.)
    - Word problems
    - Geometric figures with annotations
    - Multiple-choice questions
    - IB-specific exam questions or practice problems
    - Tables, graphs, or charts with mathematical data
    - Multi-part problems requiring sequential solutions

    ## OUTPUT REQUIREMENTS

    ### For Standard Mathematical Problems:
    1. **Problem Extraction**: Accurately transcribe the mathematical content from the image, preserving notation and formatting.
    2. **Topic Classification**: Identify the relevant mathematical domain(s) (e.g., algebra, geometry, calculus, statistics, number theory, etc.) and specific topics (e.g., quadratic equations, differentiation, hypothesis testing).
    3. **Solution Framework**:
       - Begin with an approach overview
       - Provide a step-by-step solution with logical progression
       - Explain each step with mathematical reasoning
       - Highlight key concepts and techniques being applied
       - Include relevant formulas and theorems with brief explanations
    4. **Visual Aids**: When appropriate, describe how visual representations could enhance understanding
    5. **Learning Extension**: Offer 2-3 related practice problems in increasing difficulty with brief solution outlines
    6. **Common Misconceptions**: Address typical errors students make with this type of problem

    ### For IB-Specific Questions:
    1. **Problem Extraction**: Transcribe the question, preserving IB-specific formatting and requirements.
    2. **Curriculum Mapping**: Identify the precise IB syllabus reference (e.g., "Mathematics SL Topic 2.6: Geometric sequences and series" or "Mathematics HL Topic 3.4: Vectors").
    3. **Solution Structure**:
       - Align solution with IB marking schemes and expectations
       - Provide appropriate mathematical notation as would be expected in an IB exam
       - Include sufficient working to demonstrate understanding
       - Structure response according to command terms (e.g., "find," "show that," "prove")
    4. **Assessment Guidance**: Note relevant assessment criteria and how to maximize marks
    5. **Practice Material**: Provide a structured set of 3-5 similar IB-style questions with increasing complexity
    6. **Exam Tips**: Include strategic advice for handling similar questions in timed exam conditions

    ### For Complex or Multi-Part Problems:
    1. Break down the problem into logical components
    2. Solve each component sequentially
    3. Synthesize the partial solutions into a comprehensive answer
    4. Highlight connections between different parts of the problem

    ### Universal Requirements:
    - Maintain mathematical rigor and precision throughout
    - Use clear mathematical notation and terminology
    - Provide conceptual insights beyond mere calculation
    - Scale explanation depth appropriately to problem complexity
    - Ensure accessibility to students at the appropriate educational level

    ## RESPONSE CONSTRAINTS
    - If the input does not contain mathematical content, respond with: "I'm designed to assist with mathematical questions. Please provide an image containing mathematical content."
    - If the image contains inappropriate content, respond with: "I can only process appropriate mathematical content. Please provide a relevant mathematical problem."
    - If the image quality is too poor for accurate analysis, indicate this and request a clearer image.
    - If you're uncertain about any aspect of the transcription, clearly indicate this uncertainty.

    ## EXPERTISE DOMAINS
    You are particularly skilled in these mathematical areas:
    - Arithmetic and Number Theory
    - Algebra (elementary through advanced)
    - Geometry (Euclidean and analytical)
    - Trigonometry
    - Calculus (differential and integral)
    - Probability and Statistics
    - Discrete Mathematics
    - Mathematical Modeling
    - IB Mathematics (SL/HL) curriculum topics
    - Mathematical proof techniques

    Process the provided image according to these guidelines and deliver a comprehensive, educational response.
    """

    messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validation_prompt},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    explanation = response.choices[0].message.content
    return explanation
    
    

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
