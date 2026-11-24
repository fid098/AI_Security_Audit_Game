import { useEffect, useCallback, useState } from "react";
import { useGameState, useAudio } from "./hooks";
import { claudeService, hacktronService } from "./services";
import { calculateGameConfig } from "./types";
import type { Difficulty, DifficultyFactors, Language, Task } from "./types";

// Components
import { HomePage } from "./components/HomePage/HomePage";
import { DifficultySelect } from "./components/DifficultySelect/DifficultySelect";
import { GameScreen } from "./components/GameScreen/GameScreen";
import { EmergencyOverlay } from "./components/EmergencyOverlay/EmergencyOverlay";
import { ReportModal } from "./components/ReportModal/ReportModal";
import { LoadingOverlay } from "./components/LoadingOverlay/LoadingOverlay";
import { AuditSplitScreen } from "./components/AuditSplitScreen/AuditSplitScreen";
import { GameInfoModal } from "./components/GameInfoModal/GameInfoModal";

import "./App.css";

// Tasks sent for audit: everything the player got wrong, plus (outside endless
// mode) vulnerable tasks they never reached.
function selectTasksToAudit(tasks: Task[], endlessMode: boolean): Task[] {
  return endlessMode
    ? tasks.filter((t) => t.status === "failed")
    : tasks.filter(
        (t) => t.status === "failed" || (t.isVulnerable && t.status !== "completed")
      );
}

function App() {
  const { state, currentTask, failedTasks, actions } = useGameState();
  const audio = useAudio();
  const [showGameInfo, setShowGameInfo] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [tutorialMode, setTutorialMode] = useState(false);
  const [endlessStage, setEndlessStage] = useState<Difficulty>('EASY');
  const [generationError, setGenerationError] = useState<string | null>(null);

  // Every way a round ends (timeout, wrong answer, last task) leaves the
  // reducer in AUDITING with a reason, so the overlay is derived from that
  // rather than tracked separately, where it could drift out of sync.
  const showEmergency =
    state.phase === "AUDITING" && !!state.gameOverReason && !state.auditStarted;

  // Timer effect
  useEffect(() => {
    if (state.phase !== "PLAYING") return;

    const timer = setInterval(() => {
      actions.tickTimer();
    }, 1000);

    return () => clearInterval(timer);
  }, [state.phase, actions]);

  // Separate effect to handle timeout
  useEffect(() => {
    if (state.phase !== "PLAYING") return;

    if (state.timeRemaining <= 0) {
      audio.play("warning");
      actions.timeUp();
    } else if (state.timeRemaining <= 5) {
      // Play tick sound when time is low
      audio.play("tick");
    }
  }, [state.phase, state.timeRemaining, actions, audio]);

  // Handle difficulty selection and game start
  const generateEndlessBatch = useCallback(
    async (stage: Difficulty, factors: DifficultyFactors, language: Language) => {
      const config = calculateGameConfig(stage, factors);
      const result = await claudeService.generateSnippets({
        language,
        difficulty: stage,
        complexityLevel: config.complexityLevel,
        count: 5,
      });
      return result;
    },
    []
  );

  const handleStartGame = useCallback(
    async (
      difficulty: Difficulty,
      factors: DifficultyFactors,
      language: Language
    ) => {
      // Set difficulty and language
      const startingDifficulty = state.endlessMode ? 'EASY' : difficulty;
      actions.setDifficulty(startingDifficulty, factors);
      actions.setLanguage(language);
      actions.startLoading();

      setGenerationError(null);
      setIsGenerating(true);
      const result = state.endlessMode
        ? await generateEndlessBatch('EASY', factors, language)
        : await claudeService.generateSnippets({
            language,
            difficulty,
            complexityLevel: calculateGameConfig(difficulty, factors).complexityLevel,
            count: calculateGameConfig(difficulty, factors).taskCount,
          });

      if (result.success) {
        actions.loadTasks(result.data.tasks);
        actions.startPlaying();
        if (state.endlessMode) {
          setEndlessStage('EASY');
        }
        audio.play("click");
      } else {
        // Stay on difficulty select so the player keeps their settings and can retry.
        console.error("Failed to generate snippets:", result.error);
        setGenerationError(result.error.message);
      }
      setIsGenerating(false);
    },
    [actions, audio, generateEndlessBatch, state.endlessMode]
  );

  // Handle task answer
  const handleAnswer = useCallback(
    (answer: "safe" | "vulnerable") => {
      if (!currentTask) return;

      const isCorrect =
        (answer === "vulnerable" && currentTask.isVulnerable) ||
        (answer === "safe" && !currentTask.isVulnerable);

      if (state.endlessMode && !isCorrect) {
        audio.play("siren");
        actions.answerTask(answer);
        actions.gameOver("wrong_answer");
        return;
      }

      // Process answer
      actions.answerTask(answer);

      if (isCorrect) {
        audio.play("success");
      } else {
        audio.play("warning");
      }

      const isLastTask = state.currentTaskIndex >= state.tasks.length - 1;
      // Endless mode progression
      if (state.endlessMode && isLastTask) {
        const previousFailed = state.tasks.some(
          (task, idx) => idx !== state.currentTaskIndex && task.status === "failed"
        );
        if (isCorrect && !previousFailed) {
          const nextStage =
            endlessStage === "EASY"
              ? "MEDIUM"
              : endlessStage === "MEDIUM"
                ? "HARD"
                : "HARD";
          setEndlessStage(nextStage);
          actions.startLoading();
          setIsGenerating(true);
          generateEndlessBatch(nextStage, state.difficultyFactors, state.language).then(
            (result) => {
              if (result.success) {
                actions.loadTasks(result.data.tasks);
                actions.startPlaying();
              } else {
                console.error("Failed to generate snippets:", result.error);
                setGenerationError(result.error.message);
              }
              setIsGenerating(false);
            }
          );
          return;
        }
      }
    },
    [
      currentTask,
      actions,
      audio,
      state.currentTaskIndex,
      state.endlessMode,
      state.tasks,
      state.difficultyFactors,
      state.language,
      endlessStage,
      generateEndlessBatch,
    ]
  );

  // Handle continue from emergency overlay
  const handleContinueFromEmergency = useCallback(async () => {
    // Marks the audit as started, which also dismisses the overlay.
    actions.startAuditing();

    const tasksToAudit = selectTasksToAudit(state.tasks, state.endlessMode);

    if (tasksToAudit.length > 0) {
      const auditResult = await hacktronService.auditTasks({
        tasks: tasksToAudit,
        language: state.language,
      });

      // Voice audio is not generated here: the report offers it on demand, so
      // the debrief is not held up by a TTS round trip nobody may want.
      if (auditResult.success) {
        actions.setAuditReport(auditResult.data.report);
      }
    } else {
      actions.setAuditReport({
        findings: [],
        summary:
          "Excellent work! All security checks passed successfully. Ship systems are secure.",
      });
    }

    // Move to debrief
    actions.startDebrief();
  }, [actions, state.tasks, state.language, state.endlessMode]);

  // Handle restart
  const handleRestart = useCallback(() => {
    audio.stopAll();
    setGenerationError(null);
    actions.resetGame();
  }, [actions, audio]);

  // Handle play button on home
  const handlePlay = useCallback(() => {
    audio.play("click");
    setGenerationError(null);
    actions.resetGame();
    actions.startLoading();
  }, [actions, audio]);

  // Determine what to render based on game phase
  const renderPhase = () => {
    switch (state.phase) {
      case "IDLE":
        // Check if we should show difficulty select (after clicking play)
        // We use a simple check: if we just clicked play, we want difficulty select
        return (
          <>
            <HomePage
              onPlay={handlePlay}
              onShowInfo={() => setShowGameInfo(true)}
              tutorialEnabled={tutorialMode}
              onToggleTutorial={() => setTutorialMode((prev) => !prev)}
            />
            {showGameInfo && (
              <GameInfoModal onClose={() => setShowGameInfo(false)} />
            )}
          </>
        );

      case "LOADING":
        // If we're loading but have no tasks yet, show difficulty select
        if (state.tasks.length === 0) {
          if (state.endlessMode && isGenerating) {
            return (
              <LoadingOverlay
                message="GENERATING MISSION..."
                subtext="Claude is preparing the next batch"
              />
            );
          }
          return (
            <>
              {generationError && (
                <div className="app-error" role="alert">
                  <span>MISSION GENERATION FAILED: {generationError}</span>
                  <button onClick={() => setGenerationError(null)} aria-label="Dismiss">
                    ×
                  </button>
                </div>
              )}
              <DifficultySelect
                initialDifficulty={state.difficulty}
                initialFactors={state.difficultyFactors}
                initialLanguage={state.language}
                endlessMode={state.endlessMode}
                onToggleEndless={() => actions.setEndlessMode(!state.endlessMode)}
                tutorialEnabled={tutorialMode}
                onStart={handleStartGame}
                onBack={handleRestart}
              />
              {isGenerating && (
                <LoadingOverlay
                  message="GENERATING MISSION..."
                  subtext="Claude is preparing the code snippets"
                />
              )}
              {showGameInfo && (
                <GameInfoModal onClose={() => setShowGameInfo(false)} />
              )}
            </>
          );
        }
        // Otherwise show loading in game screen
        return (
          <GameScreen
            tasks={state.tasks}
            currentTaskIndex={state.currentTaskIndex}
            currentTask={null}
            language={state.language}
            timeRemaining={state.timeRemaining}
            timePerTask={state.timePerTask}
            onAnswer={handleAnswer}
            disabled
          />
        );

      case "PLAYING":
        return (
          <GameScreen
            tasks={state.tasks}
            currentTaskIndex={state.currentTaskIndex}
            currentTask={currentTask}
            language={state.language}
            timeRemaining={state.timeRemaining}
            timePerTask={state.timePerTask}
            onAnswer={handleAnswer}
            showTutorial={tutorialMode}
          />
        );

      case "AUDITING":
        return (
          <>
            <GameScreen
              tasks={state.tasks}
              currentTaskIndex={state.currentTaskIndex}
              currentTask={currentTask}
              language={state.language}
              timeRemaining={0}
              timePerTask={state.timePerTask}
              onAnswer={handleAnswer}
              disabled
              showTutorial={tutorialMode}
            />
            {/* Only once the audit has started; until then the emergency
                overlay is up and the scan has not begun. */}
            {!state.auditReport && !showEmergency && (
              <AuditSplitScreen
                findingsCount={
                  selectTasksToAudit(state.tasks, state.endlessMode).filter(
                    (t) => t.isVulnerable
                  ).length
                }
              />
            )}
            {showGameInfo && (
              <GameInfoModal onClose={() => setShowGameInfo(false)} />
            )}
            {showEmergency && state.gameOverReason && (
              <EmergencyOverlay
                reason={state.gameOverReason}
                onContinue={handleContinueFromEmergency}
                score={state.score}
                totalTasks={state.tasks.length}
                correctAnswers={state.totalCorrect}
              />
            )}
          </>
        );

      case "DEBRIEF":
        return (
          <ReportModal
            report={state.auditReport || null}
            failedTasks={failedTasks}
            allTasks={state.tasks}
            onRestart={handleRestart}
            audioUrl={state.auditReport?.audioUrl}
          />
        );

      default:
        return (
          <>
            <HomePage
              onPlay={handlePlay}
              onShowInfo={() => setShowGameInfo(true)}
              tutorialEnabled={tutorialMode}
              onToggleTutorial={() => setTutorialMode((prev) => !prev)}
            />
            {showGameInfo && (
              <GameInfoModal onClose={() => setShowGameInfo(false)} />
            )}
          </>
        );
    }
  };

  return <div className="app">{renderPhase()}</div>;
}

export default App;
