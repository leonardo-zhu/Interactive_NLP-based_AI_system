# Interactive_NLP-based_AI_system
A dialogue-based system in Python which allows a user to complete a transaction (see below) with an AI, commonly known as a chatbot, or a conversational AI. 

### 1. Introduction

The Travel Booking Chatbot is designed to facilitate users in planning their travels by providing services such as booking flights, answering queries about travel destinations, and modifying existing travel plans. Built with Python, this chatbot leverages natural language processing (NLP) techniques to understand and respond to user queries effectively.

### 2. Chatbot Architecture

#### Functionality

The chatbot supports various functions crucial for travel planning:
- Booking Flights: Users can request to book flights between destinations on specific dates. 
- Changing Flight Dates: Allows users to modify their existing flight bookings. 
- Checking Flight Availability: Users can inquire about available flights on specific dates to different destinations.

#### Implementation

The implementation involves several Python files:
- main.py: Acts as the entry point of the chatbot, handling interactions with users and integrating various components like intent recognition and response generation. 
- intents.py: Defines the IntentClassifier class which uses a logistic regression model for intent classification. The classifier is trained on text data transformed by CountVectorizer and TfidfTransformer to identify user intents.
- train.py: Contains code for training the intent classifier with example dialogues and user queries. 
- qa.py: Manages question-answering capabilities, leveraging the trained models to respond to user queries.

#### Justification

The logistic regression model was chosen for intent classification due to its effectiveness in handling linear relationships within textual data. The use of TF-IDF for feature extraction helps in emphasizing important words and reducing the impact of frequently occurring less informative words.

#### Illustrations

(Here, you can include flowcharts or diagrams of the system architecture and interactions. Use tools like draw.io to create these visual aids.)

### 3. Conversational Design

#### Prompt Design

The chatbot employs conversational markers and structured prompts to guide the interaction flow and make the conversation feel more natural. For example:
- Acknowledgments: Simple phrases like “Got it,” “Alright,” and “Thank you” are used to acknowledge user inputs.
- Feedback: Positive feedback such as “Good job” is used to encourage continued interaction.
- Error Prompts: When an error is detected, the chatbot uses friendly reprompts like “I’m sorry, I didn’t catch that. Could you repeat?”

#### Context Tracking

The chatbot maintains context over multiple turns, allowing users to build upon previous interactions without needing to repeat themselves. For instance, if a user asks about flights to Paris and then follows up with a query about the weather there, the chatbot understands that “there” refers to Paris.

#### Personalisation

Where possible, the chatbot personalizes interactions by using the user’s name and tailoring responses based on the user’s past preferences and interactions. This approach enhances the user experience by making interactions feel more individualized and attentive.

#### Error Handling

The chatbot is designed to handle errors gracefully by providing helpful suggestions or asking clarifying questions. This helps to maintain the flow of conversation and reduces user frustration. For example, if the chatbot fails to understand a request, it might say, “I’m not sure I understood. Did you mean to ask about flight options?”

#### Confirmation

To minimize errors and ensure user satisfaction, the chatbot employs a mix of explicit and implicit confirmation techniques depending on the context:
- Explicit Confirmations: For critical actions like booking a flight, the chatbot confirms the details with the user before proceeding.
- Implicit Confirmations: For less critical information, such as answering general queries, confirmations are woven into the response to reassure the user without requiring additional input.

#### Discoverability

The chatbot is designed to help users discover and explore its capabilities through guided prompts and suggestions. It informs users about available commands and features at appropriate moments, enhancing usability and engagement.


### 4. Evaluation

The evaluation involved usability testing with a group of potential users who were asked to perform tasks like booking a flight or changing a booking. The participants provided feedback on the ease of use, understanding of the chatbot’s responses, and overall satisfaction.

### 5. Discussion

The evaluation results indicated that while users appreciated the chatbot’s ability to quickly process requests and provide pertinent information, some found the conversational flow to be limited in handling more complex queries. Reflecting on these findings, future improvements could include implementing more sophisticated NLP techniques and expanding the training dataset to cover a broader range of conversational scenarios.