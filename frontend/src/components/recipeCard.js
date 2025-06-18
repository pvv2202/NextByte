import React from 'react'
import {motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router'

function RecipeCard({id, title, img, date}) {
    const navigate = useNavigate()
    
    return (
    <motion.button 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5}}
        whileHover={{ scale: 1.08, cursor: "pointer" }} 
        className='flex w-64 max-h-90 flex-col gap-y-10 bg-white shadow-md shadow-sky-200 rounded-xl'
        onClick={() => navigate(`/landing/my-recipes/${id}`)}
    >
        <div className='w-full bg-sky-200 p-2'>
            <span>{title}</span>
        </div> 
        <div className='w-full flex justify-center p-2'>
            <div className='w-40 object-contain border-2'>
                {!img ? <img src='../nextbyte.png'/> : ''}
            </div>
        </div>
        
        <div className='text-xs flex justify-end p-2'>
            Created: {date}
        </div>
    </motion.button>
    )
}

export default RecipeCard