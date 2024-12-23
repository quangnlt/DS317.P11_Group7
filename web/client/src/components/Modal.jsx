import React from "react";
import Bad from "./Bad";
import Good from "./Good";

const Modal = ({ predict, setIsOpen }) => {
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        backgroundColor: "rgba(0,0,0,0.5)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          background: "#fff",
          padding: "20px",
          borderRadius: 8,
          minWidth: "400px",
          textAlign: "center",
        }}
      >
        <h5>Điểm trung bình của học kì của bạn là:</h5>
        <h2>{Math.round(predict * 10) / 10}</h2>
        <button
          style={{
            paddingTop: 8,
            paddingBottom: 8,
            paddingLeft: 16,
            paddingRight: 16,
            borderRadius: 8,
            border: "none",
            cursor: "pointer",
            marginBottom: 8,
            backgroundColor: "#0041d9",
          }}
          onClick={() => {
            setIsOpen(false);
          }}
        >
          <p style={{ fontWeight: "bold", color: "#fff" }}>OK</p>
        </button>
        {predict < 5 && <Bad />}
        {predict >= 5 && <Good />}
      </div>
    </div>
  );
};

export default Modal;
