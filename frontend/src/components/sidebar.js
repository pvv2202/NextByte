import { faBook, faGear, faKitchenSet, faUserGroup } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import React from 'react'
import { useState, useEffect } from 'react';
import SidebarBtn from './sidebarBtn';

function Sidebar({sidebarHidden}) {


    return (

        <div className={`sidebar text-white transition-transform duration-100 ease-in-out fixed top-0 left-0 h-screen w-40 p-2 bg-sky-950 ${sidebarHidden ?'-translate-x-full': 'translate-x-0'}`}>
            <h2 className='font-semibold text-lg'>NextByte</h2>
            <ul className='flex flex-col relative top-8 gap-y-4 w-full '>
                <li className='flex gap-x-2 items-center'>
                    <p>My Recipes</p>
                    <FontAwesomeIcon icon={faBook} />
                </li>
                <li className='flex gap-x-2 items-center'>
                    <p>Kitchen Share</p>
                    <FontAwesomeIcon icon={faUserGroup} />
                </li>
                <li className='flex gap-x-2 items-center'>
                    <p>Account</p>
                    <FontAwesomeIcon icon={faGear} />
                </li>
            </ul>
        </div>
        

    )
}

export default Sidebar