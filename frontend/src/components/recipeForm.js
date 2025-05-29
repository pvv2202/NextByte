import React from 'react'
import {useForm} from 'react-hook-form'

function RecipeForm({register, handleSubmit, onSubmit}) {

    return (
        <form onSubmit={handleSubmit(onSubmit)}>
            <input type='text' {...register("recipeTitle", {required: true,
                maxLength: 30})} 
                placeholder='Your recipe title...'
                className='p-2 rounded '
            />
            <button type='submit' className='rounded curser:hand hover:scale-105 text-white  ml-3 p-2 bg-gray-800'>Chef up</button>
        </form>
    )
}

export default RecipeForm