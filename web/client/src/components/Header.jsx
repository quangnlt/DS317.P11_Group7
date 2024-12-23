import React from "react";
import uit from "../assets/images/uit.png";

const Header = () => {
  return (
    <div
      style={{
        width: "100%",
        height: "80px",
        backgroundColor: "#fff",
        borderBottomRightRadius: 40,
        borderBottomLeftRadius: 40,
        boxShadow: "0px 5px 6px -1px rgba(137, 137, 137, 0.2)",
        marginBottom: "20px",
        display: "flex",
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingLeft: 50,
        paddingRight: 150,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "center",
          alignItems: "center",
          gap: 4,
        }}
      >
        <img src={uit} alt="Hình logo Trường" style={{ width: 60 }} />
        <h5
          style={{
            fontWeight: "bold",
            color: "#0041d9",
          }}
        >
          Trường đại học
          <br style={{ textTransform: "uppercase" }} />
          Công nghệ Thông Tin
        </h5>
      </div>
      <h5
        style={{
          fontWeight: "bold",
          color: "#0041d9",
          fontSize: 16,
        }}
      >
        Khai phá dữ liệu trong doanh nghiệp - DS317
      </h5>
      <h5
        style={{
          fontWeight: "bold",
          letterSpacing: 1,
          color: "#0041d9",
        }}
      >
        Nhóm 7
      </h5>
    </div>
  );
};

export default Header;
