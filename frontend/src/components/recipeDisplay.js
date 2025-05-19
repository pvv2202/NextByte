import React from 'react'
import { TypeAnimation } from 'react-type-animation'

function RecipeDisplay({recipe, setRecipe}) {
  return (
    <div className='w-3/4'>
        <TypeAnimation 
            sequence={[recipe, 1000]} 
            speed={100}
        />
    </div>
  )
}

export default RecipeDisplay