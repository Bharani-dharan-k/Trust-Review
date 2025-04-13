import React from 'react';

const ReviewDetector = () => {
  return (
    <div style={{ width: '100%', height: '100vh' }}>
      <iframe
        src="http://localhost:8502"
        title="Streamlit Review Authenticator"
        style={{
          width: '100%',
          height: '100%',
          border: 'none'
        }}
      />
    </div>
  );
};

export default ReviewDetector;
