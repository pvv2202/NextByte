import React from 'react'
import { Navigate, Route, Routes, Outlet, useNavigate } from 'react-router';


function Workspace({sidebarHidden, user}) {
  const navigate = useNavigate()
 
  return (
    <div className={`workspace flex justify-center h-full bg-gray-800 w-full transition-all duration-100 p-0 ${sidebarHidden ? 'ml-0' : 'ml-40'}`}>
        <Outlet />
    </div>
  )
}

export default Workspace