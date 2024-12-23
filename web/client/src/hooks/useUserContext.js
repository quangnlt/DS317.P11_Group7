import React, { createContext, useState, useContext } from "react";

// Tạo Context
const UserContext = createContext();

// Cung cấp Context cho các component con
export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null); // state lưu thông tin người dùng
  const [isLoggedIn, setIsLoggedIn] = useState(false); // state lưu trạng thái đăng nhập

  return (
    <UserContext.Provider value={{ user, setUser, isLoggedIn, setIsLoggedIn }}>
      {children}
    </UserContext.Provider>
  );
};

// Hook để sử dụng context trong các component khác
export const useUserContext = () => useContext(UserContext);
