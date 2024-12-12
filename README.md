# Interactive_NLP-based_AI_system
A Python-based dialogue system enabling users to do transactions with an AI, typically referred to as a chatbot or conversational AI. 

### 1. Introduction

The Travel Booking Chatbot is intended to assist users in organising their trips by offering services such as flight reservations, addressing enquiries regarding travel destinations, and altering current travel arrangements. This chatbot, developed in Python, utilises natural language processing (NLP) techniques to comprehend and address customer enquiries efficiently.

### 2. Chatbot Architecture

#### Functionality

The chatbot facilitates multiple essential functions for travel planning:
- Booking Flights: Users may request to reserve flights between destinations on specific dates.  
- Changing Flight Dates: Allows Permits users to alter their current flight reservations.  
- Verifying Flight Availability: Users may request information regarding available flights on designated dates to various destinations.

#### Implementation

The implementation involves several Python files:
- main.py: Serves as the chatbot's entry point, managing user interactions and integrating components such as intent recognition and response creation. 
- intents.py: Specifies the IntentClassifier class that employs a logistic regression model for intent classification. The classifier is trained on text input processed by CountVectorizer and TfidfTransformer to discern user intents.
- train.py: Includes code for training the intent classifier with sample dialogues and user enquiries. 
- qa.py: Oversees question-answering functionalities, utilising the trained models to address customer enquiries.

#### Justification

The logistic regression model was selected for intent categorisation because of its efficacy in managing linear relationships in textual data. The application of TF-IDF for feature extraction accentuates significant terms while diminishing the influence of commonly occurring, less informative words.


#### Illustrations

(Here, you can include flowcharts or diagrams of the system architecture and interactions. Use tools like draw.io to create these visual aids.)

### 3. Conversational Design

#### Prompt Design

The chatbot utilises conversational indicators and structured prompts to facilitate the interaction flow and enhance the naturalness of the dialogue. For instance:
- Acknowledgements: Concise expressions such as “Understood,” “Affirmative,” and “Thank you” are employed to recognise user contributions.
- Feedback: Affirmative remarks like “Good job” serve to promote ongoing engagement.
- Error Prompts: Upon detecting an error, the chatbot employs amicable reprompts such as “I’m sorry, I didn’t catch that.” “Could you reiterate?”

#### Context Tracking

The chatbot preserves context over several exchanges, enabling users to elaborate on prior encounters without redundancy. For example, if a customer enquires about flights to Paris and thereafter asks about the weather, the chatbot comprehends that “there” pertains to Paris.

#### Personalisation

The chatbot personalises encounters by utilising the user's name and customising responses according to the user's previous preferences and engagements. This method improves the user experience by rendering interactions more personalised and attentive.

#### Error Handling

The chatbot is engineered to manage blunders adeptly by offering constructive ideas or posing clarifying enquiries. This facilitates conversational continuity and mitigates user irritation. For instance, if the chatbot does not comprehend a request, it may respond, “I’m uncertain if I understood.” Were you enquiring about flight alternatives?

#### Confirmation

To reduce errors and guarantee user pleasure, the chatbot utilises a combination of explicit and implicit confirmation methods based on the context.
- Explicit Confirmations: For essential operations such as flight reservations, the chatbot verifies the data with the user prior to proceeding.
- Implicit Confirmations: For less crucial information, such as addressing general enquiries, confirmations are included into the response to comfort the user without necessitating further input.


#### Discoverability

The chatbot is engineered to assist users in uncovering and examining its functionalities via directed prompts and recommendations. It notifies users of accessible actions and features at opportune times, hence improving usability and engagement.


### 4. Evaluation

The evaluation included usability testing with a cohort of prospective customers tasked with activities such as booking a flight or modifying a reservation. The participants offered evaluations on the usability, comprehension of the chatbot's replies, and general contentment.

### 5. Discussion

The evaluation results revealed that users valued the chatbot's efficiency in processing questions and delivering relevant information; yet, some perceived its conversational flow as inadequate for addressing more intricate enquiries. Considering these findings, future enhancements may involve the use of advanced NLP techniques and the expansion of the training dataset to encompass a wider array of conversational contexts.
