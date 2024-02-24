import React, { useState } from 'react';
import ChatComponent from './components/navbar';

function NavBar() {
  const [activeLink, setActiveLink] = useState<string>('home');

  const handleNavLinkClick = (link: string) => {
    setActiveLink(link);
  };

  const renderPageContent = () => {
    switch (activeLink) {
      case 'prompt':
        return <ChatComponent />;
      default:
        return null;
    }
  };

  return (
    <div className="container" style={{ height: '100vh' }}>
      <nav className="navbar navbar-expand-lg bg-body-tertiary">
        <div className="container-fluid">
          <a className="navbar-brand" href="/">
            PromptlyTech
          </a>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
            aria-controls="navbarNav"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav">
              <li className={`nav-item ${activeLink === 'home' ? 'active' : ''}`}>
                <a
                  className="nav-link"
                  href="#"
                  onClick={() => handleNavLinkClick('prompt')}
                >
                  Generate prompt
                </a>
              </li>
              {/* Other menu items */}
            </ul>
          </div>
        </div>
      </nav>

      <div className="row justify-content-center mt-3">
        <div className="col-md-8">{renderPageContent()}</div>
      </div>
    </div>
  );
}

export default NavBar;
