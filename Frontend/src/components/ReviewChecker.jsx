import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./ReviewChecker.css";

const ReviewChecker = () => {
  const [darkMode, setDarkMode] = useState(false);
  const navigate = useNavigate();

  return (
    <div className={`review-container ${darkMode ? "dark" : ""}`}>
      <div className="header">
        <h2>Fake Review Detector</h2>
        <button onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? "Light Mode" : "Dark Mode"}
        </button>
      </div>

      <div className="mode-toggle">
        <button onClick={() => navigate("/review-detector")}>Single Review</button>
        <button onClick={() => navigate("/multiple")}>Multiple Reviews</button>
      </div>
    </div>
  );
};

export default ReviewChecker;
