from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from utils import get_text_embedding

load_dotenv()
chunks =  ["Prince Kumar\nMaharajganj, Siwan, Bihar 841244\nContact: 620-704-3711 | Email: pk4294884@gmail.com\nLinkedIn: linkedin.com/in/prince-kumar-42b42620a | GitHub: github.com/PrinceKumar799\n\n---\n\n## Education\n- **Chandigarh University** (Aug. 2020 â€“ June 2024)\n  BE Computer Science Engineering, Mohali, Punjab\n  CGPA: 8.55\n\n- **BSBA Inter College** (May 2017 â€“ Feb. 2019)\n  Intermediate of Science, Siwan, Bihar\n  Percentage: 83\n\n- **Children Delight School** (March 2015 â€“ April 2017)\n  Higher Secondary, Saran, Bihar\n  CGPA: 9.4\n\n---\n\n## Relevant Coursework\n- Data Structures, Algorithms Analysis, Database Management\n- Artificial Intelligence, Big Data Technology\n- Operating Systems, Web Development\n\n---\n\n## Experience\n- **Walkwel Technology Private Limited** (January 2024)\n  - Reduced a Next.js application's build size by 10%.\n  - Engineered a MERN stack solution for an e-commerce site.\n  - Implemented caching mechanisms with Redis for scalability.\n  - Created a scheduling algorithm for voice call job scheduling.\n\n---\n\n## Projects\n- **ShopKing E-commerce Platform** (ReactJS, Django, PostgreSQL, Redis, Bootstrap)\n  - Built a seamless shopping experience using MERN stack architecture.\n  - Added advanced pagination and filtering for product discovery.\n  - Optimized performance with Redis caching.\n\n- **Advanced Notes Application** (ReactJS, JavaScript, Bootstrap)\n  - Designed an advanced note-taking web application.\n\n- **Bank Customer Behavior Prediction** (Python, Machine Learning)\n  - Predicted future subscriptions of bank term deposits with 89.33% accuracy.\n\n---\n\n## Technical Skills\n- **Languages**: Python, C, HTML/CSS, JavaScript, TypeScript, Scala, C++, Java, \n- **Developer Tools**: VS Code, Eclipse, Git, Android Studio, TypeORM\n- **Frameworks**: Django, NodeJS, NestJS, ExpressJS, ReactJS, MVC\n- **Databases**: PostgreSQL, MongoDB\n\n---\n\n## Academic Achievements\n- University Academic Topper 2020-21\n- 95.6 Percentile in JEE Mains 2020\n- Merit Scholarship from Government of Bihar\n- 1st Rank in Provincial Interschool Engl", 'ish Quiz (2016)\n\n---\n\n## Leadership & Extracurricular\n- Volunteered in the formation of the largest human image of a waving flag.\n- Co-ordinator for the seating management team in college cultural events.\n']
def load_vector_and_search(question_embedding):
    question_embedding = np.array(question_embedding)
    if question_embedding.ndim == 1:
        question_embedding = question_embedding.reshape(1, -1)
    # Load the FAISS index from the file
    index = faiss.read_index("faiss_index.bin")

    print("FAISS index loaded from faiss_index.bin")

    # Perform a similarity search
    D, I = index.search(question_embedding, 2)
    print("Nearest neighbor indices:", I)
    print("Distances to nearest neighbors:", D)
    return [chunks[i] for i in I.tolist()[0]]

def format_response(context,question,client):
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Instrucations for response: 1. Act as if are talking to human, 2. Do not mention context in your response
    Query: {question}
    Answer:
    """
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    chat_response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages
    )
    return (chat_response.choices[0].message.content)
    

def run_mistral(question):
    try:
        client = client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        question =question
        question_embedding = get_text_embedding(question,client)
        context = load_vector_and_search(question_embedding)
        if not context:
            return "Sorry I have no idea regarding that"
        return format_response(context,question,client)
    except:
        return "Please try again"
