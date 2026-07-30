# DeepSeek + Tavily Fullstack LangGraph Research Agent

This project demonstrates a fullstack research application using React and LangGraph. DeepSeek generates search queries, reflects on knowledge gaps, and synthesizes the final report. Tavily retrieves current web sources, which are normalized into stable source IDs and rendered as validated citations.

<img src="./app.png" title="DeepSeek and Tavily Fullstack LangGraph" alt="Research application" width="90%">

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend.
- 🧠 Powered by a LangGraph agent for advanced research and conversational AI.
- 🔍 Dynamic search query generation using DeepSeek models.
- 🌐 Integrated web research via Tavily Search.
- 🤔 Reflective reasoning to identify knowledge gaps and refine searches.
- 📄 Generates answers with citations from gathered sources.
- 🔄 Hot-reloading for both frontend and backend during development.

## Project Structure

The project is divided into two main directories:

-   `frontend/`: Contains the React application built with Vite.
-   `backend/`: Contains the LangGraph/FastAPI application, including the research agent logic.

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.11+
-   **`DEEPSEEK_API_KEY`**: API key for DeepSeek's OpenAI-compatible endpoint.
-   **`TAVILY_API_KEY`**: API key for Tavily Search.
    1.  Navigate to the `backend/` directory.
    2.  Create a file named `.env` by copying the `backend/.env.example` file.
    3.  Set both keys in `.env`:
        `DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"` and `TAVILY_API_KEY="YOUR_TAVILY_API_KEY"`.

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
pip install .
```

**Frontend:**

```bash
cd frontend
npm install
```

**3. Run Development Servers:**

**Backend & Frontend:**

```bash
make dev
```
This will run the backend and frontend development servers.    Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).

_Alternatively, you can run the backend and frontend development servers separately. For the backend, open a terminal in the `backend/` directory and run `langgraph dev`. The backend API will be available at `http://127.0.0.1:2024`. It will also open a browser window to the LangGraph UI. For the frontend, open a terminal in the `frontend/` directory and run `npm run dev`. The frontend will be available at `http://localhost:5173`._

## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It follows these steps:

<img src="./agent.png" title="Agent Flow" alt="Agent Flow" width="50%">

1.  **Generate Initial Queries:** DeepSeek generates a bounded set of search queries.
2.  **Web Research:** Tavily runs each query in parallel and returns relevant source snippets.
3.  **Reflection & Knowledge Gap Analysis:** DeepSeek determines whether the evidence is sufficient and proposes bounded follow-up searches.
4.  **Iterative Refinement:** If gaps are found or the information is insufficient, it generates follow-up queries and repeats the web research and reflection steps (up to a configured maximum number of loops).
5.  **Finalize Answer:** DeepSeek synthesizes the gathered evidence using stable source markers, which the application validates and converts into Markdown citations.

## CLI Example

For quick one-off questions you can execute the agent from the command line. The
script `backend/examples/cli_research.py` runs the LangGraph agent and prints the
final answer:

```bash
cd backend
python examples/cli_research.py "What are the latest trends in renewable energy?"
```


## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). Below is an example of how to build a Docker image that includes the optimized frontend build and the backend server and run it via `docker-compose`.

_Note: For the docker-compose.yml example you need a LangSmith API key, you can get one from [LangSmith](https://smith.langchain.com/settings)._

_Note: If you are not running the docker-compose.yml example or exposing the backend server to the public internet, you should update the `apiUrl` in the `frontend/src/App.tsx` file to your host. Currently the `apiUrl` is set to `http://localhost:8123` for docker-compose or `http://localhost:2024` for development._

**1. Build the Docker Image:**

   Run the following command from the **project root directory**:
   ```bash
   docker build -t deepseek-tavily-fullstack-langgraph -f Dockerfile .
   ```
**2. Run the Production Server:**

   ```bash
   DEEPSEEK_API_KEY=<your_deepseek_api_key> TAVILY_API_KEY=<your_tavily_api_key> LANGSMITH_API_KEY=<your_langsmith_api_key> docker-compose up
   ```

Open your browser and navigate to `http://localhost:8123/app/` to see the application. The API will be available at `http://localhost:8123`.

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface.
- [Tailwind CSS](https://tailwindcss.com/) - For styling.
- [Shadcn UI](https://ui.shadcn.com/) - For components.
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent.
- [DeepSeek](https://api-docs.deepseek.com/) - LLM for query generation, reflection, and answer synthesis.
- [Tavily](https://docs.tavily.com/) - Search and source content retrieval.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 
