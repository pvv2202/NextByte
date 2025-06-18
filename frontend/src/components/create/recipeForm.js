import { dividerClasses } from '@mui/material/Divider'
import React from 'react'
import {useForm} from 'react-hook-form'
import {motion, AnimatePresence} from 'framer-motion'

function RecipeForm({register, handleSubmit, onSubmit, generating}) {
    

    return (
        <div>
            {!generating && (
                <motion.form 
                    onSubmit={handleSubmit(onSubmit)} 
                    className='shadow-lg shadow-sky-300'
                    initial={{ scale: 0, y: 50}}
                    animate={{ scale: 1, y: 0}}
                    transition={{ duration: 0.3 }}
                >
                        <input type='text' {...register("recipeTitle", {required: true,
                            maxLength: 30})} 
                            placeholder='Your recipe title...'
                            className='p-2 rounded'
                        />
                        <motion.button
                            type='submit'
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            className='rounded curser:hand hover:scale-105 text-white  ml-3 p-2 bg-sky-950'
                        >
                            Chef up
                        </motion.button>
                </motion.form>
            )}
        </div>
        
        
    )
}

export default RecipeForm