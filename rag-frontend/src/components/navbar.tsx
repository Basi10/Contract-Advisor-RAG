import React, { useState, useRef, useEffect } from 'react';

interface Message {
  sender: string;
  message: string;
}

const ChatComponent: React.FC = () => {
  const [userInput, setUserInput] = useState<string>('');
  const [conversation, setConversation] = useState<Message[]>([]);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUserInput(e.target.value);
  };

  const handleSendClick = () => {
    if (userInput.trim() !== '') {
      // Add user message to the conversation
      setConversation([{ sender: 'user', message: userInput }, ...conversation]);
      // Clear the input field
      setUserInput('');
    }
  };

  useEffect(() => {
    // Scroll to the bottom of the chat container when conversation updates
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversation]);

  return (
    <div className="container mt-5" style={{ position: 'relative', minHeight: '100vh' }}>
      {/* Render the conversation in reverse order */}
      <div
        className="chat-container"
        ref={chatContainerRef}
        style={{ maxHeight: '70vh', overflowY: 'hidden', paddingBottom: '50px' }}
      >
        {conversation.map((item, index) => (
          <div
            key={index}
            className={`mb-2 ${item.sender === 'user' ? 'text-end' : 'text-start'}`}
          >
            <span className={`badge bg-${item.sender === 'user' ? 'primary' : 'secondary'}`}>
              {item.sender}
            </span>{' '}
            {item.message}
          </div>
        ))}
      </div>
  
      {/* User input and send button */}
      <div className="fixed-bottom">
        <div className="row">
          <div className="col-9">
            <input
              type="text"
              value={userInput}
              onChange={handleInputChange}
              className="form-control"
              placeholder="Type your message..."
            />
          </div>
          <div className="col-3">
            <button onClick={handleSendClick} className="btn btn-dark btn-sm w-50">
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
  
};

export default ChatComponent;
