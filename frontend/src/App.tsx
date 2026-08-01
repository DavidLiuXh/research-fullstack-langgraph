import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Button } from "@/components/ui/button";

interface ResearchDimension {
  id: string;
  title: string;
  scope: string;
}

interface ResearchSource {
  title?: string;
}

interface DimensionResult {
  dimension?: ResearchDimension;
}

interface GraphUpdateEvent {
  generate_research_dimensions?: { research_dimensions?: ResearchDimension[] };
  generate_query?: { search_query?: string[] };
  web_research?: { sources_gathered?: ResearchSource[] };
  reflection?: object;
  research_dimension?: { dimension_results?: DimensionResult[] };
  finalize_answer?: object;
}

interface ResearchCustomEvent {
  type: string;
  message?: string;
  dimensions?: ResearchDimension[];
  dimension?: ResearchDimension;
  queries?: string[];
  query?: string;
  attempt?: number;
  error?: string;
  source_count?: number;
  is_sufficient?: boolean;
  knowledge_gap?: string;
  loops?: number;
}

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    reasoning_model: string;
  }>({
    apiUrl:
      import.meta.env.VITE_LANGGRAPH_API_URL ||
      (import.meta.env.DEV ? "http://localhost:2024" : window.location.origin),
    assistantId: "agent",
    messagesKey: "messages",
    onUpdateEvent: (event: GraphUpdateEvent) => {
      let processedEvent: ProcessedEvent | null = null;
      if (event.generate_research_dimensions) {
        const dimensions =
          event.generate_research_dimensions?.research_dimensions || [];
        processedEvent = {
          title: "Planning Research Dimensions",
          data:
            dimensions.map((dimension) => dimension.title).join(", ") ||
            "Research dimensions created",
        };
      } else if (event.generate_query) {
        processedEvent = {
          title: "Generating Search Queries",
          data: event.generate_query?.search_query?.join(", ") || "",
        };
      } else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const uniqueLabels = [
          ...new Set(sources.map((source) => source.title).filter(Boolean)),
        ];
        const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
        processedEvent = {
          title: "Web Research",
          data: `Gathered ${numSources} sources. Related to: ${
            exampleLabels || "N/A"
          }.`,
        };
      } else if (event.reflection) {
        processedEvent = {
          title: "Reflection",
          data: "Analysing Web Research Results",
        };
      } else if (event.research_dimension) {
        const result = event.research_dimension?.dimension_results?.[0];
        processedEvent = {
          title: "Dimension Research Complete",
          data: result?.dimension?.title || "A research dimension was completed",
        };
      } else if (event.finalize_answer) {
        processedEvent = {
          title: "Finalizing Answer",
          data: "Composing and presenting the final answer.",
        };
        hasFinalizeEventOccurredRef.current = true;
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
    onCustomEvent: (data: unknown) => {
      if (!data || typeof data !== "object" || !("type" in data)) return;
      const event = data as ResearchCustomEvent;
      let processedEvent: ProcessedEvent | null = null;
      switch (event.type) {
        case "planning_dimensions":
          processedEvent = {
            title: "Planning Research Dimensions",
            data: event.message,
          };
          break;
        case "dimensions_created":
          processedEvent = {
            title: "Research Dimensions Created",
            data: event.dimensions
              ?.map((dimension) => dimension.title)
              .join(", "),
          };
          break;
        case "queries_generated":
          processedEvent = {
            title: `Generating Queries: ${event.dimension?.title || "Dimension"}`,
            data: event.queries?.join(", ") || "",
          };
          break;
        case "search_started":
          processedEvent = { title: "Web Research", data: event.query };
          break;
        case "search_retrying":
          processedEvent = {
            title: "Retrying Web Research",
            data: `${event.query} (attempt ${(event.attempt ?? 0) + 1})`,
          };
          break;
        case "search_failed":
          processedEvent = {
            title: "Web Research Failed",
            data: `${event.query}: ${event.error}`,
          };
          break;
        case "search_completed":
          processedEvent = {
            title: "Web Research Complete",
            data: `Gathered ${event.source_count} sources for ${event.query}`,
          };
          break;
        case "reflection_completed":
          processedEvent = {
            title: `Reflection: ${event.dimension?.title || "Dimension"}`,
            data: event.is_sufficient
              ? "Evidence is sufficient"
              : event.knowledge_gap,
          };
          break;
        case "dimension_completed":
          processedEvent = {
            title: "Dimension Research Complete",
            data: `${event.dimension?.title || "Dimension"} (${event.loops} loops)`,
          };
          break;
        case "finalizing_answer":
          processedEvent = {
            title: "Finalizing Answer",
            data: "Synthesizing all research dimensions.",
          };
          hasFinalizeEventOccurredRef.current = true;
          break;
      }
      if (processedEvent) {
        setProcessedEventsTimeline((previous) => [...previous, processedEvent!]);
      }
    },
    onError: (streamError: unknown) => {
      setError(
        streamError instanceof Error ? streamError.message : String(streamError)
      );
    },
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      setError(null);
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // convert effort to, initial_search_query_count and max_research_loops
      // low means max 1 loop and 1 query
      // medium means max 3 loops and 3 queries
      // high means max 10 loops and 5 queries
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
          {error ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="flex flex-col items-center justify-center gap-4">
                <h1 className="text-2xl text-red-400 font-bold">Error</h1>
                <p className="text-red-400">{error}</p>

                <Button
                  variant="destructive"
                  onClick={() => window.location.reload()}
                >
                  Retry
                </Button>
              </div>
            </div>
          ) : thread.messages.length === 0 && !thread.isLoading ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={thread.messages}
              isLoading={thread.isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
            />
          )}
      </main>
    </div>
  );
}
