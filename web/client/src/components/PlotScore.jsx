import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Cấu hình cho Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PlotScore = ({ user }) => {
  // Hàm lọc dữ liệu
  const getFilteredData = (user) => {
    const years = [
      "Học kỳ 1",
      "Học kỳ 2",
      "Học kỳ 3",
      "Học kỳ 4",
      "Học kỳ 5",
      "Học kỳ 6",
      "Học kỳ 7",
      "Học kỳ 8",
    ];

    const scores = [
      user.diemtbhk_1,
      user.diemtbhk_2,
      user.diemtbhk_3,
      user.diemtbhk_4,
      user.diemtbhk_5,
      user.diemtbhk_6,
      user.diemtbhk_7,
      user.diemtbhk_8,
    ];

    // Lọc dữ liệu: chỉ giữ lại các giá trị khác 0
    const filteredYears = [];
    const filteredScores = [];

    years.forEach((year, index) => {
      if (scores[index] > 0) {
        filteredYears.push(year);
        filteredScores.push(scores[index]);
      }
    });

    return { filteredYears, filteredScores };
  };

  // Gọi hàm lọc dữ liệu
  const { filteredYears, filteredScores } = getFilteredData(user);

  // Dữ liệu và cấu hình biểu đồ
  const data = {
    labels: filteredYears, // Các năm học
    datasets: [
      {
        label: "Điểm trung bình các học kỳ",
        data: filteredScores, // Điểm của từng năm
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Biểu đồ điểm trung bình học kỳ",
        font: {
          size: 24, // Kích thước font chữ tiêu đề
          weight: "bold", // Đậm chữ
        },
        color: "#333", // Màu chữ tiêu đề
      },
    },
  };

  return <Line data={data} options={options} />;
};

export default PlotScore;
