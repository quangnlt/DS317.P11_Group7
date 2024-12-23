import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUserContext } from "../hooks/useUserContext";

const LogIn = () => {
  const [mssv, setMssv] = useState("");
  const [loading, setLoading] = useState(false);
  const { setUser, setIsLoggedIn } = useUserContext();
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!mssv.trim()) {
      alert("Vui lòng nhập mã số sinh viên.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ mssv_login: mssv }),
      });

      if (!response.ok) {
        throw new Error("Mã số sinh viên không tồn tại hoặc lỗi hệ thống.");
      }

      const data = await response.json();
      setUser(data);
      setIsLoggedIn(true);
      console.log("Data being passed to Prediction:", data);
      navigate("/home");
    } catch (error) {
      alert(error.message || "Đã xảy ra lỗi trong quá trình đăng nhập.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        backgroundColor: "#000",
      }}
    >
      <div
        style={{
          backgroundColor: "#f7fafc",
          width: "25%",
          minHeight: "35%",
          padding: 20,
          borderRadius: 10,
        }}
      >
        <h2 style={{ color: "#000", marginBottom: 8 }}>Đăng nhập</h2>
        <label
          htmlFor="mssv"
          style={{ color: "#000", fontSize: 15, fontWeight: "bold" }}
        >
          Mã số sinh viên
        </label>
        <input
          type="text"
          placeholder="Nhập vào mã số sinh viên"
          id="mssv"
          value={mssv}
          onChange={(e) => setMssv(e.target.value)}
          style={{
            outline: "none",
            padding: 10,
            width: "90%",
            borderRadius: 8,
            backgroundColor: "#CCD0CF",
            marginTop: 8,
            border: "none",
          }}
        />
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: 20,
          }}
        >
          <button
            onClick={handleLogin}
            disabled={loading}
            style={{
              padding: "10px 20px",
              backgroundColor: loading ? "#888" : "#0041d9",
              color: "#fff",
              fontWeight: "bold",
              border: "none",
              borderRadius: 5,
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Đang xử lý..." : "Đăng nhập"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default LogIn;
