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
uv sync --group dev
```

The Makefile explicitly places `backend/src` first on `PYTHONPATH` so the
src-layout package is importable even when LangGraph CLI is installed globally.
A project-local `uv` environment is still recommended for reproducible
dependency versions.

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

_Alternatively, run the servers separately. For the backend, open a terminal in
`backend/` and run `PYTHONPATH=src langgraph dev`. The explicit `PYTHONPATH`
ensures the src-layout package is available to a globally installed CLI.
The backend API is available at
`http://127.0.0.1:2024`. For the frontend, run `npm run dev` in `frontend/`._

## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/research_agent/graph.py`. It follows these steps:

<img src="./agent.png" title="Agent Flow" alt="Agent Flow" width="50%">

1.  **Plan Research Dimensions:** DeepSeek decomposes the main topic into distinct, complementary dimensions.
2.  **Run Dimension Subgraphs:** One isolated LangGraph subgraph runs in parallel for every dimension.
3.  **Generate Queries:** Each subgraph generates searches for its dimension and latest knowledge gap.
4.  **Web Research:** Tavily runs the queries in parallel and returns relevant source snippets.
5.  **Reflect and Refine:** DeepSeek evaluates each dimension independently. If evidence is insufficient, the knowledge gap returns to query generation for another loop.
6.  **Finalize Answer:** DeepSeek synthesizes all completed dimensions using stable source markers, which the application validates and converts into Markdown citations.

The parent graph defaults to three research dimensions. Override this with
`NUMBER_OF_RESEARCH_DIMENSIONS` (between 2 and 8). Each dimension subgraph has
its own query list, evidence, sources, reflection result, and loop counter.
Subgraph progress is emitted as custom LangGraph stream events so the frontend
continues to show activity while dimensions run. Tavily searches retry twice by
default and individual query failures degrade to partial research instead of
aborting the whole graph.

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

_Note: Development connects to `http://localhost:2024` by default. Production
uses the page origin. Set `VITE_LANGGRAPH_API_URL` at frontend build time when
the API is hosted on a different origin._

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
