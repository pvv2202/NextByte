import React from 'react'
import NextByteBot from './bot'
import CookingAnimation from './cooking-animation'
import RecipeDisplay from './recipeDisplay';

function CreateDisplay({generating, recipe, setRecipe}) {
    let content;
    if (generating){
        content = <CookingAnimation />
    } else if (recipe != null){
        content = <RecipeDisplay recipe={recipe} setRecipe={setRecipe}/>
    } else {
        content = <NextByteBot />
    }
    return (
    <div className='flex justify-center w-3/4'>
        {content}
    </div>
    )
}

export default CreateDisplay