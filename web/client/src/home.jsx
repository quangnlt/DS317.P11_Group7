import React, { useState } from "react";
import { useUserContext } from "./hooks/useUserContext";
import Header from "./components/Header";
import axios from "axios";
import Modal from "./components/Modal";
import PlotScore from "./components/PlotScore";

const Home = () => {
  const { user } = useUserContext();

  const [isOpen, setIsOpen] = useState(false);
  const [predict, setPredict] = useState(undefined);
  const [weight, setWeight] = useState("");

  const handlePredict = () => {
    const student_data = {
      namsinh: user.namsinh,
      gioitinh: user.gioitinh === "Male" ? 1 : 0,
      dtb_toankhoa: user.dtb_toankhoa,
      dtb_tichluy: user.dtb_tichluy,
      sotc_tichluy: user.sotc_tichluy,
      diemtbhk_1: user.diemtbhk_1,
      diemtbhk_2: user.diemtbhk_2,
      diemtbhk_3: user.diemtbhk_3,
      diemtbhk_4: user.diemtbhk_4,
      diemtbhk_5: user.diemtbhk_5,
      diemtbhk_6: user.diemtbhk_6,
      diemtbhk_7: user.diemtbhk_7,
      diemtbhk_8: user.diemtbhk_8,
    };
    axios
      .post("http://localhost:5000/predict", student_data)
      .then((response) => {
        console.log(response.data);
        setPredict(response.data.result);
        setWeight(response.data.coefficients);
      })
      .catch((error) => {
        console.error(
          "Lỗi khi gửi dữ liệu:",
          error.response ? error.response.data : error.message
        );
      });
  };
  console.log(weight);
  return (
    <>
      <Header />
      <div
        style={{
          width: "100%",
          height: "80vh",
          padding: 20,
        }}
      >
        <h3
          style={{
            textTransform: "uppercase",
            textAlign: "center",
            marginBottom: 12,
          }}
        >
          Bảng điểm sinh viên
        </h3>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginBottom: 15,
          }}
        >
          <div style={{}}>
            <p>Giới tính:</p>
            <p>Điểm rèn luyện:</p>
            <p>Điểm trung bình toàn khoa:</p>
            <p>Số tín chỉ tích lũy:</p>
            <p>Điểm trung bình tích lũy:</p>
          </div>
          <div>
            <p style={{ color: "#0041d9", fontWeight: "bold" }}>
              {user.gioitinh === 0 ? "Nam" : "Nữ"}
            </p>
            <p style={{ color: "#0041d9", fontWeight: "bold" }}>
              {Math.round(user.drl)}
            </p>
            <p style={{ color: "#0041d9", fontWeight: "bold" }}>
              {user.dtb_toankhoa}
            </p>
            <p style={{ color: "#0041d9", fontWeight: "bold" }}>
              {user.sotc_tichluy}
            </p>
            <p style={{ color: "#0041d9", fontWeight: "bold" }}>
              {user.dtb_tichluy}
            </p>
          </div>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <table
            style={{
              border: "1px solid #000",
              borderCollapse: "collapse",
              width: "80%",
            }}
          >
            <thead
              style={{
                backgroundColor: "#0041d9",
                color: "#fff",
                fontWeight: 500,
                textAlign: "center",
              }}
            >
              <tr>
                <th style={{ border: "1px solid black", padding: 6 }}>
                  Học kỳ
                </th>
                <th style={{ border: "1px solid black" }}>Học kỳ 1</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 2</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 3</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 4</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 5</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 6</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 7</th>
                <th style={{ border: "1px solid black" }}>Học kỳ 8</th>
              </tr>
            </thead>
            <tbody
              style={{
                textAlign: "center",
              }}
            >
              <tr>
                <td style={{ border: "1px solid black", padding: 6 }}>
                  Điểm trung bình
                </td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_1}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_2}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_3}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_4}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_5}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_6}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_7}</td>
                <td style={{ border: "1px solid black" }}>{user.diemtbhk_8}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: 20,
          }}
        >
          <button
            onClick={() => {
              handlePredict();
              console.log("hello");
              console.log(predict);
              setIsOpen(true);
            }}
            style={{
              padding: "10px 20px",
              backgroundColor: "#0041d9",
              color: "#fff",
              fontWeight: "bold",
              border: "none",
              borderRadius: 5,
              cursor: "pointer",
            }}
          >
            Dự đoán điểm
          </button>
        </div>
        {/* Hiển thị biểu đồ */}
        <div
          style={{
            margin: "20px auto",
            width: "80%",
            padding: "10px",
            backgroundColor: "#fff",
            borderRadius: 10,
            boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
          }}
        >
          <PlotScore user={user} />
        </div>
      </div>
      {isOpen && predict !== undefined && (
        <Modal predict={predict} setIsOpen={setIsOpen} weight={weight} />
      )}
    </>
  );
};

export default Home;
