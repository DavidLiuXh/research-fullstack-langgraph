import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Button } from "@/components/ui/button";
import { detectUiLanguage, translate, type UiLanguage } from "@/lib/language";

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

interface ResearchState extends Record<string, unknown> {
  messages: Message[];
  language_reference?: string;
  initial_search_query_count: number;
  max_research_loops: number;
  dimension_reflection_soft_limit: number;
  max_dimension_reflections: number;
  reasoning_model: string;
}

interface DimensionReviewInterrupt {
  type: "research_dimension_review";
  research_run_id: string;
  dimensions: ResearchDimension[];
  message: string;
}

interface TopicClarificationInterrupt {
  type: "research_topic_clarification";
  message: string;
  ambiguities: string[];
  questions: string[];
  assumptions: string[];
  reason: string;
}

interface GraphUpdateEvent {
  initialize_research_topic?: { language_reference?: string };
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
  approved?: boolean;
  feedback?: string;
  needs_clarification?: boolean;
  ambiguities?: string[];
  questions?: string[];
  assumptions?: string[];
  action?: string;
  response?: string;
  candidate_count?: number;
  accepted_count?: number;
  supplementary_count?: number;
  rejected_count?: number;
  claim_count?: number;
  passes?: boolean;
  revision_count?: number;
  gaps?: Array<{ gap_id?: string; question?: string }>;
  gap?: {
    gap_id?: string;
    question?: string;
    status?: string;
    attempt_count?: number;
  };
  assessment?: {
    gap_id?: string;
    has_progress?: boolean;
    new_matched_source_ids?: string[];
  };
  route?: string;
  strategy_level?: number;
  strategy?: string;
  merged_gap_ids?: string[];
  completion_status?: string;
  termination_reason?: string;
  planning_mode?: string;
  section_count?: number;
  thesis?: string;
}

const formatCompletionStatus = (status: string | undefined, reason: string | undefined, t: (key: string) => string) => {
  if (status === "completed_with_limitations" || status === "budget_exhausted") {
    const detail = reason ? `: ${t(reason)}` : "";
    return `${t("completed with limitations")}${detail}`;
  }
  if (status === "search_unavailable") return t("web search unavailable");
  return t(status || "complete");
};

const THREAD_STORAGE_KEY = "research-agent-thread-id";

export default function App() {
  const languageReferenceRef = useRef("");
  const [language, setLanguage] = useState<UiLanguage>("en");
  const t = useCallback((key: string, values: Record<string, string | number> = {}) =>
    translate(detectUiLanguage(languageReferenceRef.current), key, values), []);

  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [dimensionFeedback, setDimensionFeedback] = useState("");
  const [topicClarificationResponse, setTopicClarificationResponse] =
    useState("");
  const [threadId, setThreadId] = useState<string | null>(() =>
    window.localStorage.getItem(THREAD_STORAGE_KEY)
  );
  const [isRestoringThread, setIsRestoringThread] = useState(
    () => window.localStorage.getItem(THREAD_STORAGE_KEY) !== null
  );
  const thread = useStream<
    ResearchState,
    {
      InterruptType:
        | DimensionReviewInterrupt
        | TopicClarificationInterrupt;
    }
  >({
    apiUrl:
      import.meta.env.VITE_LANGGRAPH_API_URL ||
      (import.meta.env.DEV ? "http://localhost:2024" : window.location.origin),
    assistantId: "agent",
    messagesKey: "messages",
    threadId,
    onThreadId: (createdThreadId) => {
      setThreadId(createdThreadId);
      window.localStorage.setItem(THREAD_STORAGE_KEY, createdThreadId);
    },
    onUpdateEvent: (event: GraphUpdateEvent) => {
      if (event.initialize_research_topic?.language_reference) {
        languageReferenceRef.current = event.initialize_research_topic.language_reference;
        setLanguage(detectUiLanguage(languageReferenceRef.current));
      }
      let processedEvent: ProcessedEvent | null = null;
      if (event.generate_research_dimensions) {
        const dimensions =
          event.generate_research_dimensions?.research_dimensions || [];
        processedEvent = {
          title: t("Planning Research Dimensions"),
          data:
            dimensions.map((dimension) => dimension.title).join(", ") ||
            t("Research dimensions created"),
        };
      } else if (event.generate_query) {
        processedEvent = {
          title: t("Generating Search Queries"),
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
          title: t("Web Research"),
          data: t("Gathered {count} sources. Related to: {labels}.", { count: numSources, labels: exampleLabels || "—" }),
        };
      } else if (event.reflection) {
        processedEvent = {
          title: t("Reflection"),
          data: t("Analysing Web Research Results"),
        };
      } else if (event.research_dimension) {
        const result = event.research_dimension?.dimension_results?.[0];
        processedEvent = {
          title: t("Dimension Research Complete"),
          data: result?.dimension?.title || t("A research dimension was completed"),
        };
      } else if (event.finalize_answer) {
        processedEvent = {
          title: t("Finalizing Answer"),
          data: t("Composing and presenting the final answer."),
        };
        hasFinalizeEventOccurredRef.current = true;
      }
      if (processedEvent) {
        processedEvent.language = detectUiLanguage(languageReferenceRef.current);
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
        case "topic_analyzed":
          processedEvent = {
            title: t("Analyzing Research Topic"),
            data: event.needs_clarification
              ? event.ambiguities?.join(", ")
              : t("The research topic is clear."),
          };
          break;
        case "topic_clarification_received":
          processedEvent = {
            title: t("Research Topic Clarified"),
            data:
              event.action === "accept_assumptions"
                ? t("Accepted the proposed assumptions.")
                : event.response,
          };
          break;
        case "planning_dimensions":
          processedEvent = {
            title: t("Planning Research Dimensions"),
            data: event.message,
          };
          break;
        case "dimensions_created":
          processedEvent = {
            title: t("Research Dimensions Created"),
            data: event.dimensions
              ?.map((dimension) => dimension.title)
              .join(", "),
          };
          break;
        case "dimensions_reviewed":
          processedEvent = {
            title: event.approved
              ? t("Research Dimensions Approved")
              : t("Research Dimensions Rejected"),
            data: event.approved ? t("Research can begin.") : event.feedback,
          };
          break;
        case "initial_gaps_planned":
          processedEvent = {
            title: t("Planning Evidence Gaps: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data:
              event.gaps
                ?.map((gap) => gap.question || gap.gap_id)
                .filter(Boolean)
                .join(", ") || t("Initial evidence gaps planned."),
          };
          break;
        case "gap_planning_fallback":
          processedEvent = {
            title: t("Gap Planning Fallback: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Structured gap planning was unavailable; the full dimension scope was retained as a conservative gap."),
          };
          break;
        case "gap_selected":
          processedEvent = {
            title: t("Researching Gap: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: event.gap?.question || event.gap?.gap_id || t("Evidence gap selected."),
          };
          break;
        case "queries_generated":
          processedEvent = {
            title: t("Generating Queries: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: event.queries?.join(", ") || "",
          };
          break;
        case "query_generation_fallback":
          processedEvent = {
            title: t("Query Generation Fallback: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Structured query generation was unavailable; a deterministic gap-specific query was used."),
          };
          break;
        case "search_started":
          processedEvent = { title: t("Web Research"), data: event.query };
          break;
        case "search_retrying":
          processedEvent = {
            title: t("Retrying Web Research"),
            data: t("{query} (attempt {attempt})", { query: event.query || "", attempt: (event.attempt ?? 0) + 1 }),
          };
          break;
        case "search_failed":
          processedEvent = {
            title: t("Web Research Failed"),
            data: `${event.query}: ${event.error}`,
          };
          break;
        case "search_completed":
          processedEvent = {
            title: t("Web Research Complete"),
            data: t("Gathered {count} sources for {query}", { count: event.source_count || 0, query: event.query || "" }),
          };
          break;
        case "sources_evaluated":
          processedEvent = {
            title: t("Evaluating Sources: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("{accepted} accepted, {supplementary} supplementary, {rejected} rejected from {candidates} candidates", { accepted: event.accepted_count || 0, supplementary: event.supplementary_count || 0, rejected: event.rejected_count || 0, candidates: event.candidate_count || 0 }),
          };
          break;
        case "source_evaluation_fallback":
          processedEvent = {
            title: t("Source Evaluation Fallback: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Structured source scoring was unavailable; conservative fallback scoring was applied."),
          };
          break;
        case "gap_evidence_assessed":
          processedEvent = {
            title: t("Assessing Gap Evidence: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: event.assessment?.has_progress
              ? t("{count} new direct sources matched", { count: event.assessment.new_matched_source_ids?.length || 0 })
              : t("No direct evidence gain in this pass."),
          };
          break;
        case "gap_evidence_assessment_fallback":
          processedEvent = {
            title: t("Gap Evidence Fallback: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Structured evidence matching was unavailable; the gap remained open conservatively."),
          };
          break;
        case "gap_status_updated":
          processedEvent = {
            title: t("Gap Status: {status}", { status: t(event.gap?.status || event.route || "updated") }),
            data: t("{query} (attempt {attempt})", { query: event.gap?.question || event.gap?.gap_id || t("Evidence gap"), attempt: event.gap?.attempt_count || 0 }),
          };
          break;
        case "search_replanned":
          processedEvent = {
            title: t("Replanning Search: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Strategy {level}: {strategy}", { level: event.strategy_level || 0, strategy: event.strategy || t("Escalating the search strategy.") }),
          };
          break;
        case "all_gaps_processed":
          processedEvent = {
            title: t("Evidence Gaps Processed: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Running a whole-dimension coverage audit."),
          };
          break;
        case "gap_registry_merged":
          processedEvent = {
            title: t("New Evidence Gaps: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data:
              event.merged_gap_ids?.join(", ") ||
              t("No actionable gap was added."),
          };
          break;
        case "reflection_completed":
          processedEvent = {
            title: t("Reflection: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: event.is_sufficient
              ? t("Evidence is sufficient")
              : event.knowledge_gap,
          };
          break;
        case "reflection_fallback":
          processedEvent = {
            title: t("Reflection Fallback: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("Structured reflection was unavailable; conservative follow-up research was requested."),
          };
          break;
        case "dimension_completed":
          processedEvent = {
            title: t("Dimension Research Complete"),
            data: t("{dimension} ({loops} gap searches, {status})", { dimension: event.dimension?.title || t("Dimension"), loops: event.loops || 0, status: formatCompletionStatus(event.completion_status, event.termination_reason, t) }),
          };
          break;
        case "claims_extracted":
          processedEvent = {
            title: t("Extracting Evidence Claims: {dimension}", { dimension: event.dimension?.title || t("Dimension") }),
            data: t("{count} auditable claims retained", { count: event.claim_count || 0 }),
          };
          break;
        case "report_plan_created":
          processedEvent = {
            title: t("Planning Report Narrative"),
            data: t("{count} sections organized around: {thesis}", { count: event.section_count || 0, thesis: event.thesis || t("the audited findings") }),
          };
          break;
        case "report_planning_fallback":
          processedEvent = {
            title: t("Report Planning Fallback"),
            data: t("A deterministic editorial plan preserved every audited claim."),
          };
          break;
        case "drafting_report":
          processedEvent = {
            title: t("Drafting Research Report"),
            data: t("Composing the report from audited dimension claims."),
          };
          break;
        case "report_audit_completed":
          processedEvent = {
            title: event.passes ? t("Report Audit Passed") : t("Report Revision Required"),
            data: event.passes
              ? t("Coverage, evidence, and citations passed review.")
              : t("Revising the report after audit {count}.", { count: event.revision_count || 0 }),
          };
          break;
        case "finalizing_answer":
          processedEvent = {
            title: t("Finalizing Answer"),
            data: t("Synthesizing all research dimensions."),
          };
          hasFinalizeEventOccurredRef.current = true;
          break;
        case "fact_checked_report_retained":
          processedEvent = {
            title: t("Fact-checked Report Retained"),
            data: t("The report passed factual review; remaining editorial limitations did not replace its prose."),
          };
          break;
        case "safe_report_built":
          processedEvent = {
            title: t("Partial Evidence Only"),
            data: t("A complete report did not pass factual review. The output is explicitly labelled as partial evidence."),
          };
          break;
      }
      if (processedEvent) {
        processedEvent.language = detectUiLanguage(languageReferenceRef.current);
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
    if (thread.isLoading && languageReferenceRef.current) return;
    const latestHuman = [...thread.messages].reverse().find(message => message.type === "human");
    const content = latestHuman?.content;
    const reference = thread.values.language_reference || (typeof content === "string" ? content : "");
    if (reference) {
      languageReferenceRef.current = reference;
      setLanguage(detectUiLanguage(reference));
    }
  }, [thread.isLoading, thread.messages, thread.values.language_reference]);

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
    if (!threadId || thread.history.length > 0 || thread.messages.length > 0) {
      setIsRestoringThread(false);
      return;
    }

    const timeout = window.setTimeout(() => setIsRestoringThread(false), 5000);
    return () => window.clearTimeout(timeout);
  }, [thread.history.length, thread.messages.length, threadId]);

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
      languageReferenceRef.current = submittedInputValue;
      setLanguage(detectUiLanguage(submittedInputValue));
      setError(null);
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // Convert effort to queries per pass and focused attempts per evidence gap.
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      let dimension_reflection_soft_limit = 0;
      let max_dimension_reflections = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          dimension_reflection_soft_limit = 1;
          max_dimension_reflections = 2;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 2;
          dimension_reflection_soft_limit = 3;
          max_dimension_reflections = 4;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 3;
          dimension_reflection_soft_limit = 4;
          max_dimension_reflections = 6;
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
        dimension_reflection_soft_limit,
        max_dimension_reflections,
        reasoning_model: model,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
  }, [thread]);

  const handleNewSearch = useCallback(() => {
    thread.stop();
    window.localStorage.removeItem(THREAD_STORAGE_KEY);
    setThreadId(null);
    setIsRestoringThread(false);
    setError(null);
    setDimensionFeedback("");
    setTopicClarificationResponse("");
    setProcessedEventsTimeline([]);
    setHistoricalActivities({});
    hasFinalizeEventOccurredRef.current = false;
  }, [thread]);

  const dimensionReview =
    thread.interrupt?.value?.type === "research_dimension_review"
      ? thread.interrupt.value
      : null;
  const topicClarification =
    thread.interrupt?.value?.type === "research_topic_clarification"
      ? thread.interrupt.value
      : null;

  const handleTopicClarification = useCallback(() => {
    const response = topicClarificationResponse.trim();
    if (!response) {
      setError(t("Please provide the requested clarification."));
      return;
    }
    setError(null);
    setTopicClarificationResponse("");
    thread.submit(null, {
      command: { resume: { action: "clarify", response } },
    });
  }, [thread, topicClarificationResponse, t]);

  const handleAcceptTopicAssumptions = useCallback(() => {
    setError(null);
    setTopicClarificationResponse("");
    thread.submit(null, {
      command: { resume: { action: "accept_assumptions" } },
    });
  }, [thread]);

  const handleDimensionApproval = useCallback(() => {
    setError(null);
    thread.submit(null, {
      command: { resume: { approved: true, feedback: "" } },
    });
  }, [thread]);

  const handleDimensionRevision = useCallback(() => {
    const feedback = dimensionFeedback.trim();
    if (!feedback) {
      setError(t("Please explain how the research dimensions should be revised."));
      return;
    }
    setError(null);
    setDimensionFeedback("");
    thread.submit(null, {
      command: { resume: { approved: false, feedback } },
    });
  }, [dimensionFeedback, thread, t]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
          {error && !dimensionReview && !topicClarification ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="flex flex-col items-center justify-center gap-4">
                <h1 className="text-2xl text-red-400 font-bold">{t("Error")}</h1>
                <p className="text-red-400">{error}</p>

                <Button
                  variant="destructive"
                  onClick={() => window.location.reload()}
                >
                  {t("Retry")}
                </Button>
              </div>
            </div>
          ) : isRestoringThread ? (
            <div className="flex h-full items-center justify-center text-neutral-300">
              {t("Restoring previous research...")}
            </div>
          ) : thread.messages.length === 0 &&
            !thread.isLoading &&
            !dimensionReview &&
            !topicClarification ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <div className="relative h-full">
              <ChatMessagesView
                language={language}
                messages={thread.messages}
                isLoading={thread.isLoading}
                scrollAreaRef={scrollAreaRef}
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                onNewSearch={handleNewSearch}
                liveActivityEvents={processedEventsTimeline}
                historicalActivities={historicalActivities}
              />
              {topicClarification && (
                <div className="absolute inset-0 z-20 flex items-center justify-center bg-neutral-950/80 p-4 backdrop-blur-sm">
                  <section className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl border border-neutral-600 bg-neutral-800 p-6 shadow-2xl">
                    <h2 className="text-xl font-semibold">
                      {t("Clarify Research Topic")}
                    </h2>
                    <p className="mt-2 text-sm text-neutral-300">
                      {topicClarification.message}
                    </p>
                    {topicClarification.reason && (
                      <p className="mt-3 rounded-lg bg-neutral-900 p-3 text-sm text-neutral-300">
                        {topicClarification.reason}
                      </p>
                    )}
                    <div className="mt-5 space-y-4">
                      <div>
                        <h3 className="text-sm font-medium text-neutral-100">
                          {t("What needs clarification")}
                        </h3>
                        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-neutral-300">
                          {topicClarification.ambiguities.map((ambiguity) => (
                            <li key={ambiguity}>{ambiguity}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-neutral-100">
                          {t("Questions")}
                        </h3>
                        <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-neutral-300">
                          {topicClarification.questions.map((question) => (
                            <li key={question}>{question}</li>
                          ))}
                        </ol>
                      </div>
                      {topicClarification.assumptions.length > 0 && (
                        <div>
                          <h3 className="text-sm font-medium text-neutral-100">
                            {t("Suggested assumptions")}
                          </h3>
                          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-neutral-300">
                            {topicClarification.assumptions.map((assumption) => (
                              <li key={assumption}>{assumption}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                    <label className="mt-5 block text-sm font-medium text-neutral-200">
                      {t("Your clarification")}
                    </label>
                    <textarea
                      value={topicClarificationResponse}
                      onChange={(event) =>
                        setTopicClarificationResponse(event.target.value)
                      }
                      placeholder={t("Provide the missing scope, subject, or interpretation...")}
                      className="mt-2 min-h-28 w-full resize-y rounded-lg border border-neutral-600 bg-neutral-900 p-3 text-sm outline-none focus:border-blue-400"
                      disabled={thread.isLoading}
                    />
                    {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
                    <div className="mt-5 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
                      {topicClarification.assumptions.length > 0 && (
                        <Button
                          variant="outline"
                          className="border-neutral-500 bg-neutral-700 text-neutral-100 hover:bg-neutral-600 hover:text-white"
                          onClick={handleAcceptTopicAssumptions}
                          disabled={thread.isLoading}
                        >
                          {t("Continue with Assumptions")}
                        </Button>
                      )}
                      <Button
                        className="bg-blue-600 text-white hover:bg-blue-500"
                        onClick={handleTopicClarification}
                        disabled={thread.isLoading}
                      >
                        {t("Submit Clarification")}
                      </Button>
                    </div>
                  </section>
                </div>
              )}
              {dimensionReview && (
                <div className="absolute inset-0 z-20 flex items-center justify-center bg-neutral-950/80 p-4 backdrop-blur-sm">
                  <section className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl border border-neutral-600 bg-neutral-800 p-6 shadow-2xl">
                    <h2 className="text-xl font-semibold">
                      {t("Review Research Dimensions")}
                    </h2>
                    <p className="mt-2 text-sm text-neutral-300">
                      {dimensionReview.message}
                    </p>
                    <div className="mt-5 space-y-3">
                      {dimensionReview.dimensions.map((dimension) => (
                        <div
                          key={dimension.id}
                          className="rounded-lg border border-neutral-600 bg-neutral-900 p-4"
                        >
                          <h3 className="font-medium text-neutral-100">
                            {dimension.title}
                          </h3>
                          <p className="mt-1 text-sm text-neutral-300">
                            {dimension.scope}
                          </p>
                        </div>
                      ))}
                    </div>
                    <label className="mt-5 block text-sm font-medium text-neutral-200">
                      {t("Revision feedback (required when rejecting)")}
                    </label>
                    <textarea
                      value={dimensionFeedback}
                      onChange={(event) => setDimensionFeedback(event.target.value)}
                      placeholder={t("Describe missing perspectives, unwanted overlap, or a preferred focus...")}
                      className="mt-2 min-h-28 w-full resize-y rounded-lg border border-neutral-600 bg-neutral-900 p-3 text-sm outline-none focus:border-blue-400"
                      disabled={thread.isLoading}
                    />
                    {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
                    <div className="mt-5 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
                      <Button
                        variant="outline"
                        className="border-red-400/70 bg-red-950/40 text-red-100 hover:bg-red-900/70 hover:text-white"
                        onClick={handleDimensionRevision}
                        disabled={thread.isLoading}
                      >
                        {t("Regenerate with Feedback")}
                      </Button>
                      <Button
                        className="bg-emerald-600 text-white hover:bg-emerald-500"
                        onClick={handleDimensionApproval}
                        disabled={thread.isLoading}
                      >
                        {t("Approve and Continue")}
                      </Button>
                    </div>
                  </section>
                </div>
              )}
            </div>
          )}
      </main>
    </div>
  );
}
