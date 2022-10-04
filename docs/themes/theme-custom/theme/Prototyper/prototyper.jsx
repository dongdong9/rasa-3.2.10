import React from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import ThemeContext from '@theme/_contexts/ThemeContext';
import { isProductionBuild, uuidv4 } from '@theme/_utils';

import PrototyperContext from './context';

const jsonHeaders = {
  Accept: 'application/json',
  'Content-Type': 'application/json',
};
const trackerPollingInterval = 2000;

const Prototyper = ({
  children,
  startPrototyperApi,
  trainModelApi,
  chatBlockSelector,
  chatBlockScriptUrl,
}) => {
  const [hasStarted, setHasStarted] = React.useState(false);
  const [trackingId, setTrackingId] = React.useState(null);
  const [projectDownloadUrl, setProjectDownloadUrl] = React.useState(null);
  const [trainingData, setTrainingData] = React.useState({});
  const [pollingIntervalId, setPollingIntervalId] = React.useState(null);

  const [baseUrl, setBaseUrl] = React.useState("");
  const [tracker, setTracker] = React.useState({});
  const [chatState, setChatState] = React.useState("not_trained");

  // FIXME: once we can use `rasa-ui` outside of `rasa-x`, we can remove this
  const insertChatBlockScript = () => {
    if (ExecutionEnvironment.canUseDOM) {
      const scriptElement = document.createElement('script');
      scriptElement.src = chatBlockScriptUrl;
      document.body.appendChild(scriptElement);
    }
  };

  // update tracking id when component is mounting
  React.useEffect(() => {
    setTrackingId(isProductionBuild() ? uuidv4() : 'the-hash');
    insertChatBlockScript();
    updateChatBlock();
  }, []);

  // update chat block when chatState or tracker changes
  React.useEffect(() => {
    updateChatBlock();
  }, [chatState, tracker, trainingData]);

  const clearPollingInterval = React.useCallback(() => {
    if (pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
  }, [pollingIntervalId, setPollingIntervalId]);

  const onLiveCodeStart = React.useCallback((name, value) => {
    setTrainingData((prevTrainingData) => ({...prevTrainingData, [name]: value}));
  }, [setTrainingData]);

  const onLiveCodeChange = React.useCallback((name, value) => {
    setTrainingData((prevTrainingData) => ({ ...prevTrainingData, [name]: value }));

    if (chatState === "ready") {
      clearPollingInterval();
      setChatState("needs_to_be_retrained");
      updateChatBlock();
    }

    if (!hasStarted) {
      // track the start here
      setHasStarted(true);
      fetch(startPrototyperApi, {
        method: 'POST',
        headers: jsonHeaders,
        body: JSON.stringify({
          tracking_id: trackingId,
          editor: 'main',
        }),
      });
    }
  }, [
    clearPollingInterval,
    hasStarted,
    setHasStarted,
    trackingId,
    chatState,
    setChatState,
    setTrainingData,
    jsonHeaders
  ]);

  const trainModel = (trainingData) => {
    setChatState("training");
    clearPollingInterval();

    fetch(trainModelApi, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ tracking_id: trackingId, ...trainingData }),
    })
      .then((response) => response.json())
      .then((data) => {
        setProjectDownloadUrl(data.project_download_url);
        if (data.rasa_service_url) {
          startFetchingTracker(data.rasa_service_url);
        }
      });
  };

  const downloadProject = () => {
    if (projectDownloadUrl) {
      location.href = projectDownloadUrl;
    }
  };

  const updateChatBlock = () => {
    if (!ExecutionEnvironment.canUseDOM) {
      return;
    }

    if (!window.ChatBlock) {
      setTimeout(() => updateChatBlock(baseUrl, tracker), 500);
      return;
    }

    window.ChatBlock.default.init({
      onSendMessage: (message) => {
        sendMessage(baseUrl, message);
      },
      onTrainClick: () => {
        trainModel(trainingData);
      },
      username: trackingId,
      tracker: tracker,
      selector: chatBlockSelector,
      state: chatState,
    });
  };

  const fetchTracker = (baseUrl) => {
    fetch(`${baseUrl}/conversations/${trackingId}/tracker`, {
      method: 'GET',
      header: 'jsonHeaders',
    })
      .then((response) => response.json())
      .then((tracker) => {
        setBaseUrl(baseUrl);
        setTracker(tracker);
        setChatState("ready");
      });
  };

  const sendMessage = (baseUrl, message) => {
    fetch(`${baseUrl}/webhooks/rest/webhook`, {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({
        sender: trackingId,
        message: message,
      }),
    }).then(() => {
      fetchTracker(baseUrl);
    });
  };

  const startFetchingTracker = (baseUrl) => {
    setChatState("deploying");
    fetchTracker(baseUrl);

    const updateIntervalId = setInterval(() => {
      fetchTracker(baseUrl);
    }, trackerPollingInterval);

    setPollingIntervalId(updateIntervalId);
  };

  return (
    <ThemeContext.Provider value={{ onLiveCodeChange, onLiveCodeStart }}>
      <PrototyperContext.Provider value={{ trainModel, downloadProject, chatState }}>
        {children}
      </PrototyperContext.Provider>
    </ThemeContext.Provider>
  );
};

export default Prototyper;
